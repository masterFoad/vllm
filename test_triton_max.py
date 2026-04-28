# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl


@triton.jit
def my_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    val = offset * 2
    max_val, max_id = tl.max(val, axis=0, return_indices=True)
    tl.store(out_ptr + pid, max_id)


out = torch.zeros(1, dtype=torch.int32, device="cuda")
my_kernel[(1,)](out, BLOCK_SIZE=128)
print(out)
