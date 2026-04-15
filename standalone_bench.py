import time
import torch
import numpy as np


class MockInputBuffers:
    def __init__(self, max_num_reqs, max_num_tokens, device):
        self.device = device
        self.max_num_reqs = max_num_reqs

        # New persistent buffers
        self.idx_mapping_np = np.empty(max_num_reqs, dtype=np.int32)
        self.idx_mapping = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self.cu_num_logits_np = np.empty(max_num_reqs + 1, dtype=np.int32)
        self.cu_num_logits = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.query_start_loc_np = np.empty(max_num_reqs + 1, dtype=np.int32)
        self.query_start_loc = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.expanded_idx_mapping = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.expanded_local_pos = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.arange_num_reqs = torch.arange(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.arange_num_reqs_np = np.arange(max_num_reqs + 1, dtype=np.int32)
        self.zeros_num_reqs = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


class MockModelRunner:
    def __init__(self, device):
        self.device = device
        self.max_num_reqs = 1024
        self.input_buffers = MockInputBuffers(1024, 4096, device)
        self.req_id_to_index = {f"req_{i}": i for i in range(1024)}

    def prepare_inputs_old(self, num_reqs, num_tokens_per_req, num_reqs_padded):
        # Old logic
        req_ids = [f"req_{i}" for i in range(num_reqs)]
        idx_mapping_iter = map(self.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = torch.from_numpy(idx_mapping_np).to(self.device)

        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        cu_num_logits = torch.arange(
            num_reqs + 1, device=self.device, dtype=torch.int32
        )
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )

        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_tokens_per_req, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = sum(num_tokens_per_req)

        self.input_buffers.query_start_loc.copy_(torch.from_numpy(query_start_loc_np))
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]

    def prepare_inputs_new(self, num_reqs, num_tokens_per_req, num_reqs_padded):
        # New logic
        req_ids = [f"req_{i}" for i in range(num_reqs)]

        idx_mapping_iter = map(self.req_id_to_index.get, req_ids)
        idx_mapping_np = self.input_buffers.idx_mapping_np[:num_reqs]
        for i, idx in enumerate(idx_mapping_iter):
            idx_mapping_np[i] = idx
        idx_mapping = self.input_buffers.idx_mapping[:num_reqs]
        idx_mapping.copy_(torch.from_numpy(idx_mapping_np))

        cu_num_logits_np = self.input_buffers.arange_num_reqs_np[: num_reqs + 1]
        cu_num_logits = self.input_buffers.arange_num_reqs[: num_reqs + 1]
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = self.input_buffers.zeros_num_reqs[:num_reqs]

        query_start_loc_np = self.input_buffers.query_start_loc_np[
            : num_reqs_padded + 1
        ]
        query_start_loc_np[0] = 0
        np.cumsum(num_tokens_per_req, out=query_start_loc_np[1 : num_reqs + 1])
        if num_reqs_padded > num_reqs:
            query_start_loc_np[num_reqs + 1 :] = sum(num_tokens_per_req)
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(torch.from_numpy(query_start_loc_np))


def run_bench():
    runner = MockModelRunner("cpu")

    batch_sizes = [10, 250, 10]

    # Warmup
    for bs in batch_sizes:
        tokens = np.ones(bs, dtype=np.int32)
        runner.prepare_inputs_old(bs, tokens, bs)
        runner.prepare_inputs_new(bs, tokens, bs)

    for name, func in [
        ("Old", runner.prepare_inputs_old),
        ("New", runner.prepare_inputs_new),
    ]:
        latencies = []
        for i in range(1000):
            bs = batch_sizes[i % len(batch_sizes)]
            tokens = np.ones(bs, dtype=np.int32)
            padded = bs if bs % 16 == 0 else bs + (16 - bs % 16)

            start = time.perf_counter()
            func(bs, tokens, padded)
            end = time.perf_counter()
            latencies.append((end - start) * 1_000_000)

        latencies = np.array(latencies)
        print(f"{name} Results (1000 iter):")
        print(f"p50: {np.percentile(latencies, 50):.2f} us")
        print(f"p95: {np.percentile(latencies, 95):.2f} us")
        print(f"p99: {np.percentile(latencies, 99):.2f} us")


if __name__ == "__main__":
    run_bench()
