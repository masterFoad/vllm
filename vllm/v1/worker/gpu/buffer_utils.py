# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)


def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    if x.device == out.device:
        return out.copy_(x)

    # Copy directly to GPU — explicit pin_memory() causes sporadic stalls
    # under high concurrency due to CUDA driver contention. The driver
    # handles the transfer efficiently without manual pinning.
    return out.copy_(x, non_blocking=True)


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)


class UvaBufferPool:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
    ):
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    def __init__(
        self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int = 2
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)
        self.device_tensor = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.device_tensor = self.pool.copy_to_uva(
            self.np[:n] if n is not None else self.np
        )
        return self.device_tensor


class StagedWriteTensor(ABC):
    @abstractmethod
    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        pass

    @abstractmethod
    def stage_write_elem(self, index: int, x: int) -> None:
        pass

    @abstractmethod
    def apply_write(self) -> None:
        pass

    @abstractmethod
    def clear_staged_writes(self) -> None:
        pass

    @property
    @abstractmethod
    def device_tensor(self) -> torch.Tensor:
        pass


class CUDAStagedWriteTensor(StagedWriteTensor):
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ):
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self._device_tensor = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self._device_tensor = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    @property
    def device_tensor(self) -> torch.Tensor:
        return self._device_tensor

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if x is None or (
            isinstance(x, (list, tuple, np.ndarray, torch.Tensor)) and len(x) == 0
        ):
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, self.dtype, self.device, pin_memory=True
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self._device_tensor,
            self._device_tensor.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


class CPUStagedWriteTensor(StagedWriteTensor):
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self._device_tensor = torch.zeros(size, dtype=dtype, device=device)
        self._staged_writes: list[tuple[int, int, list]] = []

    @property
    def device_tensor(self) -> torch.Tensor:
        return self._device_tensor

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        if x is None or (
            isinstance(x, (list, tuple, np.ndarray, torch.Tensor)) and len(x) == 0
        ):
            return
        # Use list.append() for fast staging
        self._staged_writes.append((index, start, list(x)))

    def stage_write_elem(self, index: int, x: int) -> None:
        self._staged_writes.append((index, 0, [x]))

    def apply_write(self) -> None:
        if not self._staged_writes:
            return
        for index, start, x in self._staged_writes:
            n = len(x)
            if self._device_tensor.ndim == 1:
                # For 1D tensors (e.g. total_len, num_computed_tokens)
                # index is the position, start is ignored
                self._device_tensor[index] = x[0]
            else:
                # For 2D tensors (e.g. all_token_ids, block_tables)
                self._device_tensor[index, start : start + n] = torch.tensor(
                    x, dtype=self.dtype, device=self.device
                )
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_writes.clear()


class DeviceMemoryManager:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cpu = device.type == "cpu"

    def get_staged_writer(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ) -> StagedWriteTensor:
        if self.is_cpu:
            return CPUStagedWriteTensor(size, dtype, self.device)
        else:
            return CUDAStagedWriteTensor(
                size, dtype, self.device, max_concurrency, uva_instead_of_gpu
            )


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )
