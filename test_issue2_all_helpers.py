import torch
import time


# --- MOCK OLD BASELINE HELPERS (which allocate internally AND do computation) ---
def old_expand_idx_mapping(
    idx_mapping, total_num_logits, cu_num_logits, max_expand_len
):
    a = torch.zeros(total_num_logits, dtype=torch.int32, device=idx_mapping.device)
    b = torch.zeros(total_num_logits, dtype=torch.int32, device=idx_mapping.device)
    expand_idx_mapping(
        idx_mapping, total_num_logits, cu_num_logits, max_expand_len, a, b
    )
    return a, b


def old_combine_sampled_and_draft_tokens(
    input_ids,
    idx_mapping,
    last_sampled,
    start_loc,
    seq_lens,
    prefill_len,
    draft,
    cu_num_logits,
    num_logits,
):
    c = torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
    combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled,
        start_loc,
        seq_lens,
        prefill_len,
        draft,
        cu_num_logits,
        num_logits,
        c,
    )
    return c


def old_get_num_sampled_and_rejected(
    num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
):
    c = torch.empty_like(num_sampled)
    get_num_sampled_and_rejected(
        num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len, c
    )
    return num_sampled, c


# --- REAL OPTIMIZED HELPERS ---
from vllm.v1.worker.gpu.input_batch import (
    expand_idx_mapping,
    combine_sampled_and_draft_tokens,
    get_num_sampled_and_rejected,
    InputBuffers,
)


def run_tests():
    device = torch.device("cuda")
    max_num_reqs = 256
    max_num_tokens = 4096

    # Setup Persistent Buffers
    buffers = InputBuffers(max_num_reqs, max_num_tokens, device)

    # Setup mock data for step 1 (Large Batch: 200 reqs)
    n_large = 200
    idx_mapping_large = torch.arange(n_large, dtype=torch.int32, device=device)
    cu_num_logits_large = torch.arange(n_large + 1, dtype=torch.int32, device=device)
    num_sampled_large = torch.ones(n_large, dtype=torch.int32, device=device)
    seq_lens_large = torch.full((n_large,), 10, dtype=torch.int32, device=device)
    prefill_len = torch.full((max_num_reqs,), 0, dtype=torch.int32, device=device)

    # Fill buffers with garbage data from large batch
    buffers.num_rejected[:n_large].fill_(99)
    buffers.logits_indices[:n_large].fill_(99)
    buffers.expanded_idx_mapping[:n_large].fill_(99)

    # --- CORRECTNESS TEST: Large Batch -> Small Batch (No Stale Data) ---
    print("--- CORRECTNESS TEST ---")
    n_small = 5
    idx_mapping_small = torch.arange(n_small, dtype=torch.int32, device=device)
    cu_num_logits_small = torch.arange(n_small + 1, dtype=torch.int32, device=device)
    num_sampled_small = torch.ones(n_small, dtype=torch.int32, device=device)
    seq_lens_small = torch.full((n_small,), 10, dtype=torch.int32, device=device)
    input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
    last_sampled = torch.ones(max_num_reqs, dtype=torch.int32, device=device)
    draft_tokens = torch.empty((max_num_reqs, 0), dtype=torch.int32, device=device)
    query_start_loc = torch.arange(n_small + 1, dtype=torch.int32, device=device)

    # Call optimized helpers for small batch
    _, num_rejected_out = get_num_sampled_and_rejected(
        num_sampled_small,
        seq_lens_small,
        cu_num_logits_small,
        idx_mapping_small,
        prefill_len,
        buffers.num_rejected,
    )

    logits_indices_out = combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping_small,
        last_sampled,
        query_start_loc,
        seq_lens_small,
        prefill_len,
        draft_tokens,
        cu_num_logits_small,
        n_small,
        buffers.logits_indices,
    )

    expand_a, expand_b = expand_idx_mapping(
        idx_mapping_small,
        n_small,
        cu_num_logits_small,
        1,
        buffers.expanded_idx_mapping,
        buffers.expanded_local_pos,
    )

    # Verify no stale data leaked from large batch
    # We expect valid indices and zeros (from logic), not the '99' we filled earlier
    print(f"num_rejected shape: {num_rejected_out.shape} (Expected {n_small})")
    assert num_rejected_out.shape[0] == n_small
    assert not (num_rejected_out == 99).any(), "Stale data leaked into num_rejected!"

    print(f"logits_indices shape: {logits_indices_out.shape} (Expected {n_small})")
    assert logits_indices_out.shape[0] == n_small
    assert not (logits_indices_out == 99).any(), (
        "Stale data leaked into logits_indices!"
    )

    print(f"expanded_idx_mapping shape: {expand_a.shape} (Expected {n_small})")
    assert expand_a.shape[0] == n_small
    assert not (expand_a == 99).any(), "Stale data leaked into expanded_idx_mapping!"

    print("SUCCESS: Correctness and Stale Data Leakage checks passed.\n")

    # --- INTEGRATED BENCHMARK (All 3 Helpers) ---
    print("--- INTEGRATED PREP-FLOW BENCHMARK (Batch=256) ---")
    n_bench = 256
    idx_mapping_bench = torch.arange(n_bench, dtype=torch.int32, device=device)
    cu_num_logits_bench = torch.arange(n_bench + 1, dtype=torch.int32, device=device)
    num_sampled_bench = torch.ones(n_bench, dtype=torch.int32, device=device)
    seq_lens_bench = torch.full((n_bench,), 10, dtype=torch.int32, device=device)
    query_start_loc_bench = torch.arange(n_bench + 1, dtype=torch.int32, device=device)

    # WARMUP
    for _ in range(10):
        get_num_sampled_and_rejected(
            num_sampled_bench,
            seq_lens_bench,
            cu_num_logits_bench,
            idx_mapping_bench,
            prefill_len,
            buffers.num_rejected,
        )
        combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping_bench,
            last_sampled,
            query_start_loc_bench,
            seq_lens_bench,
            prefill_len,
            draft_tokens,
            cu_num_logits_bench,
            n_bench,
            buffers.logits_indices,
        )
        expand_idx_mapping(
            idx_mapping_bench,
            n_bench,
            cu_num_logits_bench,
            1,
            buffers.expanded_idx_mapping,
            buffers.expanded_local_pos,
        )

    # 1. BASELINE TIMING
    torch.cuda.synchronize()
    start_allocs = torch.cuda.memory_stats(device).get("active.all.allocated", 0)
    start_time = time.perf_counter()
    for _ in range(1000):
        old_get_num_sampled_and_rejected(
            num_sampled_bench,
            seq_lens_bench,
            cu_num_logits_bench,
            idx_mapping_bench,
            prefill_len,
        )
        old_combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping_bench,
            last_sampled,
            query_start_loc_bench,
            seq_lens_bench,
            prefill_len,
            draft_tokens,
            cu_num_logits_bench,
            n_bench,
        )
        old_expand_idx_mapping(idx_mapping_bench, n_bench, cu_num_logits_bench, 1)

    torch.cuda.synchronize()
    old_time = time.perf_counter() - start_time
    old_allocs = (
        torch.cuda.memory_stats(device).get("active.all.allocated", 0) - start_allocs
    )

    # 2. OPTIMIZED TIMING
    torch.cuda.synchronize()
    start_allocs = torch.cuda.memory_stats(device).get("active.all.allocated", 0)
    start_time = time.perf_counter()
    for _ in range(1000):
        get_num_sampled_and_rejected(
            num_sampled_bench,
            seq_lens_bench,
            cu_num_logits_bench,
            idx_mapping_bench,
            prefill_len,
            buffers.num_rejected,
        )
        combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping_bench,
            last_sampled,
            query_start_loc_bench,
            seq_lens_bench,
            prefill_len,
            draft_tokens,
            cu_num_logits_bench,
            n_bench,
            buffers.logits_indices,
        )
        expand_idx_mapping(
            idx_mapping_bench,
            n_bench,
            cu_num_logits_bench,
            1,
            buffers.expanded_idx_mapping,
            buffers.expanded_local_pos,
        )

    torch.cuda.synchronize()
    new_time = time.perf_counter() - start_time
    new_allocs = (
        torch.cuda.memory_stats(device).get("active.all.allocated", 0) - start_allocs
    )

    print(
        f"BASELINE (Internal Allocs): {old_time * 1000:.2f} ms | GPU Allocs: {old_allocs}"
    )
    print(
        f"OPTIMIZED (Scratch Reuse):  {new_time * 1000:.2f} ms | GPU Allocs: {new_allocs}"
    )
    print(f"Total GPU Allocs Avoided:   {old_allocs - new_allocs}")


if __name__ == "__main__":
    run_tests()
