# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch
import triton
import triton.language as tl

# We will benchmark the actual old vs new paths


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos
    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        prob = tl.load(
            target_probs_ptr + token_idx * vocab_size + vocab_offset,
            mask=(vocab_mask & (vocab_offset != draft_token_id)),
            other=0.0,
        )

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)


@triton.jit
def old_rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_probs_ptr,  # [num_tokens, vocab_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    max_spec_len,
    vocab_size,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
            draft_prob = 1.0
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + token_idx)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + token_idx)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )


@triton.jit
def new_rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    uniform_probs_ptr,  # [num_tokens]
    max_spec_len,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
            draft_prob = 1.0
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + token_idx)

            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                    draft_token_id,
                )
            else:
                rejected = True

                max_val = float("-inf")
                recovered_id = 0
                for v in range(0, vocab_size, BLOCK_SIZE):
                    vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                    vocab_mask = vocab_offset < vocab_size

                    prob = tl.load(
                        target_probs_ptr + token_idx * vocab_size + vocab_offset,
                        mask=(vocab_mask & (vocab_offset != draft_token_id)),
                        other=0.0,
                    )

                    inv_q = tl.load(
                        inv_q_ptr + req_idx * vocab_size + vocab_offset,
                        mask=vocab_mask,
                        other=0.0,
                    )

                    score = prob * inv_q
                    local_max, local_id = tl.max(score, axis=0, return_indices=True)

                    if local_max > max_val:
                        max_val = local_max
                        recovered_id = v + local_id

                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                    recovered_id,
                )


def run_bench():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    batch_size = 128
    vocab_size = 128000
    max_spec_len = 8
    num_tokens = batch_size * max_spec_len

    output_token_ids = torch.zeros(
        (batch_size, max_spec_len + 1), dtype=torch.int32, device=device
    )
    cu_num_draft_tokens = torch.arange(
        max_spec_len, num_tokens + 1, max_spec_len, dtype=torch.int32, device=device
    )
    draft_token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    target_probs = torch.rand(
        (num_tokens, vocab_size), dtype=torch.float32, device=device
    )
    inv_q = torch.rand((batch_size, vocab_size), dtype=torch.float32, device=device)
    recovered_token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    # Set uniform_probs to 1.0 to force rejection at position 0
    uniform_probs_reject_0 = torch.ones(num_tokens, dtype=torch.float32, device=device)

    # Set uniform_probs to 0.0 for pos < 4, and 1.0 for pos >= 4 to force rejection at pos 4
    uniform_probs_reject_4 = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    for i in range(batch_size):
        uniform_probs_reject_4[i * max_spec_len + 4 :] = 1.0

    # Warmup
    for _ in range(10):
        sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            inv_q,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        old_rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            recovered_token_ids,
            uniform_probs_reject_0,
            max_spec_len,
            vocab_size,
        )
        new_rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            inv_q,
            uniform_probs_reject_0,
            max_spec_len,
            vocab_size,
            BLOCK_SIZE=8192,
        )

    torch.cuda.synchronize()

    # Benchmark Old (Eager computation for all 8 positions)
    start = time.perf_counter()
    for _ in range(1000):
        sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            inv_q,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        old_rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            recovered_token_ids,
            uniform_probs_reject_0,
            max_spec_len,
            vocab_size,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    old_time = (end - start) * 1000 / 1000  # ms per iter

    # Benchmark New (Rejects at position 0)
    start = time.perf_counter()
    for _ in range(1000):
        new_rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            inv_q,
            uniform_probs_reject_0,
            max_spec_len,
            vocab_size,
            BLOCK_SIZE=8192,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    new_time_pos0 = (end - start) * 1000 / 1000  # ms per iter

    # Benchmark New (Rejects at position 4)
    start = time.perf_counter()
    for _ in range(1000):
        new_rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            inv_q,
            uniform_probs_reject_4,
            max_spec_len,
            vocab_size,
            BLOCK_SIZE=8192,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    new_time_pos4 = (end - start) * 1000 / 1000  # ms per iter

    # Benchmark New (All accepted)
    uniform_probs_accept_all = torch.zeros(
        num_tokens, dtype=torch.float32, device=device
    )
    start = time.perf_counter()
    for _ in range(1000):
        new_rejection_random_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_probs,
            inv_q,
            uniform_probs_accept_all,
            max_spec_len,
            vocab_size,
            BLOCK_SIZE=8192,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    new_time_accept_all = (end - start) * 1000 / 1000  # ms per iter

    print(f"Old Eager Path (8 positions): {old_time:.4f} ms")
    print(
        f"New Lazy Path (Reject at pos 0): {new_time_pos0:.4f} ms ({old_time / new_time_pos0:.2f}x speedup)"
    )
    print(
        f"New Lazy Path (Reject at pos 4): {new_time_pos4:.4f} ms ({old_time / new_time_pos4:.2f}x speedup)"
    )
    print(
        f"New Lazy Path (All accepted): {new_time_accept_all:.4f} ms ({old_time / new_time_accept_all:.2f}x speedup)"
    )


if __name__ == "__main__":
    run_bench()
