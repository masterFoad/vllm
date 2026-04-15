import sys
from types import ModuleType

dummy_c = ModuleType("vllm._C")
dummy_c.ops = ModuleType("vllm._C.ops")
dummy_libtorch = ModuleType("vllm._C_stable_libtorch")
dummy_flashinfer = ModuleType("flashinfer")

sys.modules["vllm._C"] = dummy_c
sys.modules["vllm._C_stable_libtorch"] = dummy_libtorch
sys.modules["flashinfer"] = dummy_flashinfer

import torch
import numpy as np
from unittest.mock import MagicMock
from vllm.v1.worker.gpu.input_batch import InputBuffers, InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput


class MockReqStates:
    def __init__(self):
        self.req_id_to_index = {}
        self.any_prefills = MagicMock(return_value=False)
        self.next_prefill_tokens = MagicMock()

        mock_gpu_tensor = MagicMock()
        self.all_token_ids = MagicMock(gpu=mock_gpu_tensor)
        self.prefill_len = MagicMock(gpu=mock_gpu_tensor)
        self.num_computed_tokens = MagicMock(gpu=mock_gpu_tensor)


class MockModelRunner(GPUModelRunner):
    def __init__(self):
        self.device = torch.device("cpu")
        self.max_num_reqs = 1024
        self.max_num_tokens = 4096
        self.num_speculative_steps = 0
        self.use_dcp = False

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_num_tokens,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=False,
            vocab_size=32000,
            block_sizes=[16],
            kernel_block_sizes=[16],
        )
        self.req_states = MockReqStates()


import vllm.v1.worker.gpu.model_runner
import vllm.utils.torch_utils


def mock_async_copy_to_gpu(src, device=None, out=None, non_blocking=False):
    if out is not None:
        out.copy_(torch.from_numpy(src))
        return out
    else:
        return torch.from_numpy(src).to(device)


vllm.v1.worker.gpu.model_runner.async_copy_to_gpu = mock_async_copy_to_gpu
vllm.utils.torch_utils.async_copy_to_gpu = mock_async_copy_to_gpu


vllm.v1.worker.gpu.model_runner.prepare_prefill_inputs = MagicMock()
vllm.v1.worker.gpu.model_runner.prepare_pos_seq_lens = MagicMock()


def test_large_then_small_reuse():
    runner = MockModelRunner()

    # Large
    large_num_reqs = 100
    runner.req_states.req_id_to_index = {f"req_{i}": i for i in range(large_num_reqs)}
    num_scheduled_tokens = {f"req_{i}": 1 for i in range(large_num_reqs)}

    sched_out_large = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=large_num_reqs,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    desc_large = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=large_num_reqs, num_reqs=None
    )

    runner.prepare_inputs(sched_out_large, desc_large)

    # Small
    small_num_reqs = 5
    runner.req_states.req_id_to_index = {f"req_{i}": i for i in range(small_num_reqs)}
    num_scheduled_tokens = {f"req_{i}": 1 for i in range(small_num_reqs)}

    sched_out_small = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=small_num_reqs,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    desc_small = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=small_num_reqs, num_reqs=None
    )

    runner.prepare_inputs(sched_out_small, desc_small)

    # Verify correctness for the second (small) run
    query_start_loc = runner.input_buffers.query_start_loc[: small_num_reqs + 1]
    expected_qsl = torch.arange(small_num_reqs + 1, dtype=torch.int32)
    assert torch.equal(query_start_loc, expected_qsl), (
        "query_start_loc stale data leakage"
    )

    idx_mapping = runner.input_buffers.idx_mapping[:small_num_reqs]
    expected_idx = torch.arange(small_num_reqs, dtype=torch.int32)
    assert torch.equal(idx_mapping, expected_idx), "idx_mapping stale data leakage"

    print("test_large_then_small_reuse passed!")


def test_padding_boundary():
    runner = MockModelRunner()

    boundary = 16
    padded_tokens = 64
    runner.req_states.req_id_to_index = {f"req_{i}": i for i in range(boundary - 1)}
    num_scheduled_tokens = {f"req_{i}": 1 for i in range(boundary - 1)}

    sched_out = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=boundary - 1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=padded_tokens, num_reqs=boundary
    )

    copy_sizes = []
    original_copy = vllm.v1.worker.gpu.model_runner.async_copy_to_gpu

    def mock_copy_intercept(src, out=None, *args, **kwargs):
        if out is not None:
            copy_sizes.append((src.shape, out.shape))
        return original_copy(src, out=out, *args, **kwargs)

    vllm.v1.worker.gpu.model_runner.async_copy_to_gpu = mock_copy_intercept

    runner.prepare_inputs(sched_out, desc)

    vllm.v1.worker.gpu.model_runner.async_copy_to_gpu = original_copy

    qsl = runner.input_buffers.query_start_loc[: boundary + 1].cpu().numpy()
    assert qsl[-1] == padded_tokens, "padding boundary tail not padded to num_tokens"
    assert qsl[-2] == padded_tokens, "dummy padding element not set to num_tokens"

    print("Copy sizes intercepted:", copy_sizes)
    print("test_padding_boundary passed!")


if __name__ == "__main__":
    test_large_then_small_reuse()
    test_padding_boundary()
