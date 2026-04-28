# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import triton
import triton.language as tl


@triton.jit
def my_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # do something
    pass
