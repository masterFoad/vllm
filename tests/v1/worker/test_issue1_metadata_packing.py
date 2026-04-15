import sys
from unittest.mock import MagicMock

sys.modules["vllm._C"] = MagicMock()
sys.modules["vllm._C_stable_libtorch"] = MagicMock()
sys.modules["flashinfer"] = MagicMock()
import vllm.v1.worker.gpu.input_batch

vllm.v1.worker.gpu.input_batch._expand_idx_mapping_kernel = MagicMock()

import pytest
import numpy as np
import torch
from unittest.mock import patch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput, CachedRequestData
from vllm.config import LoadConfig
from tests.v1.worker.test_gpu_model_runner import (
    _schedule_new_request,
    initialize_kv_cache,
    dist_init,
)
import tests.v1.worker.test_gpu_model_runner as old_module


def custom_get_vllm_config():
    config = old_module.get_vllm_config()
    config.load_config = LoadConfig(load_format="dummy")
    config.model_config.enforce_eager = True
    return config


old_module.get_vllm_config = custom_get_vllm_config
from tests.v1.worker.test_gpu_model_runner import model_runner


@pytest.fixture(autouse=True)
def setup_torch():
    torch.set_default_dtype(torch.float16)


def test_large_then_small_reuse(model_runner, dist_init):
    # Prepare large batch
    large_num_reqs = 100
    req_ids_large = [f"req_large_{i}" for i in range(large_num_reqs)]
    sched_out_large = _schedule_new_request(*req_ids_large)

    # We must add these requests to model_runner.req_states
    model_runner._update_states(sched_out_large)

    # Mock BatchExecutionDescriptor
    # Note: prepare_inputs expects num_tokens_after_padding. Let's just pass num_tokens.
    desc_large = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=sched_out_large.total_num_scheduled_tokens,
        num_reqs=None,  # PIECEWISE
    )

    # First call: large
    input_batch_large = model_runner.prepare_inputs(sched_out_large, desc_large)

    # Now prepare a small batch
    small_num_reqs = 5
    req_ids_small = [f"req_small_{i}" for i in range(small_num_reqs)]
    sched_out_small = _schedule_new_request(*req_ids_small)

    model_runner._update_states(sched_out_small)

    desc_small = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=sched_out_small.total_num_scheduled_tokens,
        num_reqs=None,
    )

    # Second call: small
    input_batch_small = model_runner.prepare_inputs(sched_out_small, desc_small)

    # Assert query_start_loc contents
    expected_query_start_loc = np.zeros(small_num_reqs + 1, dtype=np.int32)
    # Each new request has 3 tokens scheduled in _schedule_new_request
    np.cumsum(
        np.full(small_num_reqs, 3, dtype=np.int32), out=expected_query_start_loc[1:]
    )

    # The actual tensor length returned in input_batch is num_reqs_padded + 1
    # which is small_num_reqs + 1 because num_reqs=None for piecewise.
    query_start_loc_gpu = model_runner.input_buffers.query_start_loc[
        : small_num_reqs + 1
    ]

    assert torch.equal(
        query_start_loc_gpu.cpu(), torch.from_numpy(expected_query_start_loc)
    )

    # Assert idx_mapping
    # It maps req_id to req_index in req_states
    idx_mapping_gpu = model_runner.input_buffers.idx_mapping[:small_num_reqs]
    expected_idx_mapping = [
        model_runner.req_states.req_id_to_index[req_id] for req_id in req_ids_small
    ]
    assert torch.equal(
        idx_mapping_gpu.cpu(), torch.tensor(expected_idx_mapping, dtype=torch.int32)
    )


def test_padding_boundary(model_runner, dist_init):
    # Test boundary behavior

    # Scenario: padding boundary = 16 (meaning num_reqs=16)
    boundary = 16
    req_ids_below = [f"req_below_{i}" for i in range(boundary - 1)]
    sched_out_below = _schedule_new_request(*req_ids_below)
    model_runner._update_states(sched_out_below)

    desc_below = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=100,  # Some padded length
        num_reqs=boundary,  # Padded to boundary
    )

    # Intercept async_copy_to_gpu calls to check sizes
    copy_sizes = []

    import vllm.v1.worker.gpu.model_runner as mr_module

    original_copy = mr_module.async_copy_to_gpu

    def mock_copy(src, out=None, *args, **kwargs):
        if out is not None:
            copy_sizes.append((src.shape, out.shape))
        return original_copy(src, out=out, *args, **kwargs)

    with patch.object(mr_module, "async_copy_to_gpu", side_effect=mock_copy):
        model_runner.prepare_inputs(sched_out_below, desc_below)

    # Check that sizes are bounded by padded length, not max length
    # query_start_loc should be copied with size num_reqs_padded + 1
    # max_num_reqs is probably 100 or something larger in tests
    assert any(src_shape == (boundary + 1,) for src_shape, out_shape in copy_sizes), (
        "query_start_loc padded copy missing"
    )
    assert not any(
        src_shape == (model_runner.max_num_reqs + 1,)
        for src_shape, out_shape in copy_sizes
    ), "copied max_num_reqs elements!"

    # query_start_loc should be padded: the tail elements should be set to num_tokens
    qsl = model_runner.input_buffers.query_start_loc[: boundary + 1].cpu().numpy()
    assert qsl[-1] == 100  # It should be padded to num_tokens (100)
    assert qsl[-2] == 100  # The dummy element right after the real ones
