# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.states import RequestState

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_input_buffers_cpu_views_share_storage() -> None:
    input_buffers = InputBuffers(
        max_num_reqs=8,
        max_num_tokens=64,
        device=torch.device("cpu"),
    )

    input_buffers.idx_mapping_np[:3] = [4, 2, 7]
    assert torch.equal(
        input_buffers.idx_mapping_cpu[:3],
        torch.tensor([4, 2, 7], dtype=torch.int32),
    )

    input_buffers.cu_num_logits_np[:4] = [0, 2, 3, 6]
    assert torch.equal(
        input_buffers.cu_num_logits_cpu[:4],
        torch.tensor([0, 2, 3, 6], dtype=torch.int32),
    )

    input_buffers.query_start_loc_np[:4] = [0, 5, 9, 12]
    assert torch.equal(
        input_buffers.query_start_loc_cpu[:4],
        torch.tensor([0, 5, 9, 12], dtype=torch.int32),
    )

    input_buffers.seq_lens_cpu_upper_bound_np[:3] = [11, 12, 13]
    assert torch.equal(
        input_buffers.seq_lens_cpu_upper_bound_cpu[:3],
        torch.tensor([11, 12, 13], dtype=torch.int32),
    )


@CUDA_ONLY
def test_prepare_pos_seq_lens_clears_padded_entries() -> None:
    max_num_reqs = 8
    input_buffers = InputBuffers(
        max_num_reqs=max_num_reqs,
        max_num_tokens=64,
        device=torch.device("cuda"),
    )

    # First run with all requests to make seq_lens non-zero everywhere.
    idx_mapping_full = torch.arange(max_num_reqs, dtype=torch.int32, device="cuda")
    query_start_loc_full = torch.arange(
        0, max_num_reqs + 1, dtype=torch.int32, device="cuda"
    )
    num_computed = torch.ones(max_num_reqs, dtype=torch.int32, device="cuda")
    prepare_pos_seq_lens(
        idx_mapping_full,
        query_start_loc_full,
        num_computed,
        input_buffers.positions,
        input_buffers.seq_lens,
    )
    assert torch.all(input_buffers.seq_lens > 0)

    # Second run with fewer requests should zero out padded seq_lens.
    num_reqs_small = 3
    idx_mapping_small = torch.arange(num_reqs_small, dtype=torch.int32, device="cuda")
    query_start_loc_small = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device="cuda")
    num_computed_small = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
    prepare_pos_seq_lens(
        idx_mapping_small,
        query_start_loc_small,
        num_computed_small,
        input_buffers.positions,
        input_buffers.seq_lens,
    )

    assert torch.all(input_buffers.seq_lens[num_reqs_small:] == 0)


@CUDA_ONLY
def test_expand_idx_mapping_uses_preallocated_buffers() -> None:
    idx_mapping = torch.tensor([2, 7, 5], dtype=torch.int32, device="cuda")
    # per-request num logits = [2, 1, 3]
    cu_num_logits = torch.tensor([0, 2, 3, 6], dtype=torch.int32, device="cuda")
    total_num_logits = int(cu_num_logits[-1].item())
    max_expand_len = 3

    expanded_idx_mapping_buf = torch.empty(16, dtype=torch.int32, device="cuda")
    expanded_local_pos_buf = torch.empty(16, dtype=torch.int32, device="cuda")

    expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
        idx_mapping,
        total_num_logits,
        cu_num_logits,
        max_expand_len,
        expanded_idx_mapping_buf,
        expanded_local_pos_buf,
    )

    assert expanded_idx_mapping.data_ptr() == expanded_idx_mapping_buf.data_ptr()
    assert expanded_local_pos.data_ptr() == expanded_local_pos_buf.data_ptr()
    assert expanded_idx_mapping.shape[0] == total_num_logits
    assert expanded_local_pos.shape[0] == total_num_logits

    expected_idx = torch.tensor([2, 2, 7, 5, 5, 5], dtype=torch.int32, device="cuda")
    expected_pos = torch.tensor([0, 1, 0, 0, 1, 2], dtype=torch.int32, device="cuda")
    assert torch.equal(expanded_idx_mapping, expected_idx)
    assert torch.equal(expanded_local_pos, expected_pos)


@CUDA_ONLY
def test_expand_idx_mapping_raises_if_buffers_too_small() -> None:
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu_num_logits = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    total_num_logits = 5
    max_expand_len = 3

    with pytest.raises(AssertionError, match="expanded_idx_mapping capacity"):
        expand_idx_mapping(
            idx_mapping,
            total_num_logits,
            cu_num_logits,
            max_expand_len,
            torch.empty(4, dtype=torch.int32, device="cuda"),
            torch.empty(5, dtype=torch.int32, device="cuda"),
        )

    with pytest.raises(AssertionError, match="expanded_local_pos capacity"):
        expand_idx_mapping(
            idx_mapping,
            total_num_logits,
            cu_num_logits,
            max_expand_len,
            torch.empty(5, dtype=torch.int32, device="cuda"),
            torch.empty(4, dtype=torch.int32, device="cuda"),
        )


def test_combine_sampled_and_draft_tokens_validates_output_capacity() -> None:
    # Assertion should fire before any Triton launch, so CPU tensors are fine.
    with pytest.raises(AssertionError, match="logits_indices capacity"):
        combine_sampled_and_draft_tokens(
            input_ids=torch.empty(1, dtype=torch.int32),
            idx_mapping=torch.empty(1, dtype=torch.int32),
            last_sampled_tokens=torch.empty(1, dtype=torch.int32),
            query_start_loc=torch.empty(2, dtype=torch.int32),
            seq_lens=torch.empty(1, dtype=torch.int32),
            prefill_len=torch.empty(1, dtype=torch.int32),
            draft_tokens=torch.empty(1, 1, dtype=torch.int32),
            cu_num_logits=torch.empty(2, dtype=torch.int32),
            num_logits=2,
            logits_indices=torch.empty(1, dtype=torch.int64),
        )


@CUDA_ONLY
def test_request_state_any_prefills_matches_expected_and_reuses_scratch() -> None:
    req_states = RequestState(
        max_num_reqs=8,
        max_model_len=32,
        max_num_batched_tokens=32,
        num_speculative_steps=2,
        vocab_size=128,
        device=torch.device("cuda"),
    )
    req_states.num_computed_prefill_tokens[:] = np.array(
        [0, 4, 3, 10, 1, 8, 2, 7], dtype=np.int32
    )
    req_states.prefill_len.np[:] = np.array(
        [1, 4, 5, 10, 1, 8, 2, 7], dtype=np.int32
    )

    active = np.array([0, 1, 2, 3], dtype=np.int32)
    assert req_states.any_prefills(active)

    smaller = np.array([4, 5, 6, 7], dtype=np.int32)
    assert not req_states.any_prefills(smaller)

    mixed = np.array([2, 5], dtype=np.int32)
    assert req_states.any_prefills(mixed)
