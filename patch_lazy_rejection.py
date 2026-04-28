# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import re

with open("vllm/v1/sample/rejection_sampler.py") as f:
    content = f.read()

# 1. Replace sample_recovered_tokens call with inv_q generation
old_sample_recovered = """    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )"""

new_inv_q = """    # Compute inv_q for recovered tokens.
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)
    inv_q = q.reciprocal()
    BLOCK_SIZE = 8192"""

content = content.replace(old_sample_recovered, new_inv_q)

# 2. Update rejection_random_sample_kernel call
old_kernel_call = """    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
    )"""

new_kernel_call = """    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        inv_q,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
    )"""

content = content.replace(old_kernel_call, new_kernel_call)

# 3. Remove sample_recovered_tokens function
# We'll just use regex to remove it
content = re.sub(
    r"def sample_recovered_tokens\(.*?\)\s*->\s*torch\.Tensor:.*?return recovered_token_ids\n",
    "",
    content,
    flags=re.DOTALL,
)

# 4. Update rejection_random_sample_kernel definition
old_kernel_def = """def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )"""

new_kernel_def = """def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
            uniform_prob = tl.load(uniform_probs_ptr + token_idx)
            if SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1.0
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr + token_idx * vocab_size + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + token_idx * vocab_size + draft_token_id
                )
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            
            if accepted:
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, draft_token_id
                )
            else:
                rejected = True
                
                max_val = float("-inf")
                recovered_id = 0
                for v in range(0, vocab_size, BLOCK_SIZE):
                    vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                    vocab_mask = vocab_offset < vocab_size

                    if NO_DRAFT_PROBS:
                        prob = tl.load(
                            target_probs_ptr + token_idx * vocab_size + vocab_offset,
                            mask=(vocab_mask & (vocab_offset != draft_token_id)),
                            other=0.0,
                        )
                    else:
                        draft_prob_v = tl.load(
                            draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                            mask=vocab_mask,
                            other=0.0,
                        )
                        target_prob_v = tl.load(
                            target_probs_ptr + token_idx * vocab_size + vocab_offset,
                            mask=vocab_mask,
                            other=0.0,
                        )
                        prob = tl.maximum(target_prob_v - draft_prob_v, 0.0)

                    inv_q = tl.load(
                        inv_q_ptr + req_idx * vocab_size + vocab_offset,
                        mask=vocab_mask,
                        other=0.0,
                    )

                    # Local tile reduction
                    score = prob * inv_q
                    local_max, local_id = tl.max(score, axis=0, return_indices=True)

                    if local_max > max_val:
                        max_val = local_max
                        recovered_id = v + local_id

                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, recovered_id
                )"""

content = content.replace(old_kernel_def, new_kernel_def)

# 5. Remove sample_recovered_tokens_kernel
content = re.sub(
    r"@triton\.jit\ndef sample_recovered_tokens_kernel\(.*?\n\n",
    "",
    content,
    flags=re.DOTALL,
)

with open("vllm/v1/sample/rejection_sampler.py", "w") as f:
    f.write(content)
