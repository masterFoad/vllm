import sys
from unittest.mock import MagicMock

sys.modules["vllm._C"] = MagicMock()
sys.modules["vllm._C_stable_libtorch"] = MagicMock()
sys.modules["flashinfer"] = MagicMock()
sys.modules["vllm.v1.worker.gpu.input_batch"]._expand_idx_mapping_kernel = MagicMock()

import time
import torch
import numpy as np
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CUDAGraphMode
from tests.v1.worker.test_gpu_model_runner import (
    model_runner,
    _schedule_new_request,
    get_vllm_config,
    initialize_kv_cache,
    dist_init,
)


def run_benchmark(runner):
    # We will simulate 1000 iterations of alternating batch sizes
    # 10, 250, 10
    batch_sizes = [10, 250, 10]

    # Warmup
    for bs in batch_sizes:
        req_ids = [f"req_warmup_{bs}_{i}" for i in range(bs)]
        sched_out = _schedule_new_request(*req_ids)
        runner._update_states(sched_out)
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE,
            num_tokens=sched_out.total_num_scheduled_tokens,
            num_reqs=None,
        )
        runner.prepare_inputs(sched_out, desc)

    torch.cuda.synchronize()

    latencies = []

    for i in range(1000):
        bs = batch_sizes[i % len(batch_sizes)]
        req_ids = [f"req_bench_{i}_{j}" for j in range(bs)]
        sched_out = _schedule_new_request(*req_ids)
        runner._update_states(sched_out)
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE,
            num_tokens=sched_out.total_num_scheduled_tokens,
            num_reqs=None,
        )

        start = time.perf_counter()
        runner.prepare_inputs(sched_out, desc)
        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1_000_000)  # microseconds

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"Benchmark Results (1000 iterations):")
    print(f"p50: {p50:.2f} us")
    print(f"p95: {p95:.2f} us")
    print(f"p99: {p99:.2f} us")


if __name__ == "__main__":
    # We don't have pytest fixture injected, so we manually instantiate it
    torch.set_default_dtype(torch.float16)

    # Setup distributed
    from tests.utils import ensure_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    import os

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=1)

    # Setup runner
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.config import set_current_vllm_config, LoadConfig

    vllm_config = get_vllm_config()
    vllm_config.load_config = LoadConfig(load_format="dummy")
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            num_heads, head_size, 0.1
        )
        runner = GPUModelRunner(vllm_config, "cuda")
        initialize_kv_cache(runner)

        run_benchmark(runner)
