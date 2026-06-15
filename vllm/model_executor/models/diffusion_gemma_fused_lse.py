# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helpers for DiffusionGemma fused vocab reductions.

This module is intentionally not wired into serving yet. It is the first
single-rank Tier-2 building block: stream ``hidden @ lm_head`` over vocab tiles,
apply DiffusionGemma final-logit softcap, and compute row-wise logsumexp without
materializing full ``[rows, vocab]`` logits.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


_DEFAULT_BLOCK_M = 32
_DEFAULT_BLOCK_N = 128
_DEFAULT_BLOCK_K = 64
_DEFAULT_NUM_WARPS = 8


@triton.jit
def _softcap_lse_partial_kernel(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    partial_max: torch.Tensor,
    partial_denom: torch.Tensor,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    softcap: tl.constexpr,
    temperature: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    vocab_block = tl.program_id(1)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    vocab_offsets = vocab_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, hidden_size, BLOCK_K):
        k = k_start + k_offsets
        hidden_tile = tl.load(
            hidden + row_offsets[:, None] * hidden_size + k[None, :],
            mask=(row_offsets[:, None] < rows) & (k[None, :] < hidden_size),
            other=0.0,
        )
        weight_tile = tl.load(
            weight + vocab_offsets[:, None] * hidden_size + k[None, :],
            mask=(vocab_offsets[:, None] < vocab_size)
            & (k[None, :] < hidden_size),
            other=0.0,
        )
        acc += tl.dot(hidden_tile, tl.trans(weight_tile))

    # Triton 3.6 in the current vLLM image does not expose tl.tanh. Use the
    # equivalent logistic form: tanh(z) = 2 / (1 + exp(-2z)) - 1.
    z = acc / softcap
    scaled = ((2.0 / (1.0 + tl.exp(-2.0 * z))) - 1.0) * softcap
    scaled = scaled / temperature
    scaled = tl.where(
        (row_offsets[:, None] < rows) & (vocab_offsets[None, :] < vocab_size),
        scaled,
        -float("inf"),
    )

    tile_max = tl.max(scaled, axis=1)
    tile_denom = tl.sum(tl.exp(scaled - tile_max[:, None]), axis=1)

    out_offsets = row_offsets * num_vocab_blocks + vocab_block
    tl.store(partial_max + out_offsets, tile_max, mask=row_offsets < rows)
    tl.store(partial_denom + out_offsets, tile_denom, mask=row_offsets < rows)


def diffusion_gemma_softcap_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float,
    temperature: float,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Compute row-wise softcapped lm-head logsumexp without full logits.

    Args:
        hidden: ``[rows, hidden_size]`` CUDA tensor.
        weight: ``[vocab_size, hidden_size]`` CUDA tensor.
        softcap: DiffusionGemma final-logit softcap value.
        temperature: Sampling temperature divisor applied after softcap.
        block_m: Triton row tile size.
        block_n: Triton vocab tile size.
        block_k: Triton reduction tile size.
        num_warps: Triton kernel warp count.

    Returns:
        ``[rows]`` fp32 logsumexp values.
    """
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight must be rank-2 tensors")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("hidden and weight must be CUDA tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if softcap <= 0:
        raise ValueError("softcap must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    rows, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    if rows == 0:
        return torch.empty((0,), device=hidden.device, dtype=torch.float32)

    num_vocab_blocks = triton.cdiv(vocab_size, block_n)
    partial_shape = (rows, num_vocab_blocks)
    partial_max = torch.empty(partial_shape, device=hidden.device,
                              dtype=torch.float32)
    partial_denom = torch.empty_like(partial_max)
    grid = (triton.cdiv(rows, block_m), num_vocab_blocks)

    _softcap_lse_partial_kernel[grid](
        hidden,
        weight,
        partial_max,
        partial_denom,
        rows,
        hidden_size,
        vocab_size,
        num_vocab_blocks,
        float(softcap),
        float(temperature),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    row_max = partial_max.max(dim=1).values
    denom = (partial_denom * torch.exp(partial_max - row_max[:, None])).sum(dim=1)
    return row_max + denom.log()
