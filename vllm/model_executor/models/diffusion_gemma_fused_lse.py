# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton helpers for DiffusionGemma fused vocab reductions.

This module is intentionally not wired into serving yet. It is a single-rank
Tier-2 building block: stream ``hidden @ lm_head`` over vocab tiles, apply
DiffusionGemma final-logit softcap, and compute row-wise sampler reductions
without materializing full ``[rows, vocab]`` logits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_DEFAULT_BLOCK_M = 32
_DEFAULT_BLOCK_N = 128
_DEFAULT_BLOCK_K = 64
# Memory/speed knob for the online soft-embedding path. On A100 with
# rows=2048, vocab=262144: 4096 keeps peak scratch near 180 MiB but is slower;
# 32768 keeps peak near 1.2 GiB and was fastest in the prototype sweep. Keep
# this explicit until serving chooses a memory-first or throughput-first policy.
_DEFAULT_SOFT_EMBED_CHUNK = 32768
_DEFAULT_NUM_WARPS = 8
_GUMBEL_CHUNK = 32768
_INT64_MIX_A = -7046029254386353131
_INT64_MIX_B = -4658895280553007687
_INT64_MIX_C = -7723592293110705685
_INT64_MASK_53 = (1 << 53) - 1
_FLOAT_2_NEG_53 = 1.0 / float(1 << 53)


@triton.jit
def _softcap_reduce_partial_kernel(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    partial_max: torch.Tensor,
    partial_denom: torch.Tensor,
    partial_expected: torch.Tensor,
    partial_argmax_token: torch.Tensor,
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

    z = acc / softcap
    scaled = libdevice.tanh(z) * softcap
    scaled = scaled / temperature
    valid = (row_offsets[:, None] < rows) & (vocab_offsets[None, :] < vocab_size)
    scaled = tl.where(valid, scaled, -float("inf"))

    tile_max = tl.max(scaled, axis=1)
    weights = tl.exp(scaled - tile_max[:, None])
    tile_denom = tl.sum(weights, axis=1)
    scaled_for_expected = tl.where(valid, scaled, 0.0)
    tile_expected = tl.sum(weights * scaled_for_expected, axis=1)

    max_token_candidates = tl.where(
        scaled == tile_max[:, None], vocab_offsets[None, :], vocab_size
    )
    tile_argmax_token = tl.min(max_token_candidates, axis=1)

    out_offsets = row_offsets * num_vocab_blocks + vocab_block
    row_mask = row_offsets < rows
    tl.store(partial_max + out_offsets, tile_max, mask=row_mask)
    tl.store(partial_denom + out_offsets, tile_denom, mask=row_mask)
    tl.store(partial_expected + out_offsets, tile_expected, mask=row_mask)
    tl.store(partial_argmax_token + out_offsets, tile_argmax_token, mask=row_mask)


def diffusion_gemma_softcap_lse_entropy_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float,
    temperature: float,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute softcapped lm-head LSE, entropy, and greedy argmax.

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
        Tuple of ``[rows]`` fp32 logsumexp, ``[rows]`` fp32 entropy, and
        ``[rows]`` int64 greedy-token ids.
    """
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight must be rank-2 tensors")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("hidden and weight must be CUDA tensors")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if softcap <= 0:
        raise ValueError("softcap must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    rows, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    if vocab_size == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        empty = torch.empty((0,), device=hidden.device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=hidden.device, dtype=torch.int64)
        return empty, empty, empty_tokens

    num_vocab_blocks = triton.cdiv(vocab_size, block_n)
    partial_shape = (rows, num_vocab_blocks)
    partial_max = torch.empty(partial_shape, device=hidden.device,
                              dtype=torch.float32)
    partial_denom = torch.empty_like(partial_max)
    partial_expected = torch.empty_like(partial_max)
    partial_argmax_token = torch.empty(partial_shape, device=hidden.device,
                                       dtype=torch.int64)
    grid = (triton.cdiv(rows, block_m), num_vocab_blocks)

    _softcap_reduce_partial_kernel[grid](
        hidden,
        weight,
        partial_max,
        partial_denom,
        partial_expected,
        partial_argmax_token,
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
    rescale = torch.exp(partial_max - row_max[:, None])
    denom = (partial_denom * rescale).sum(dim=1)
    expected = (partial_expected * rescale).sum(dim=1) / denom
    lse = row_max + denom.log()
    entropy = lse - expected

    argmax_block = partial_max.max(dim=1).indices
    argmax_tokens = partial_argmax_token.gather(
        1, argmax_block[:, None]
    ).squeeze(1)
    return lse, entropy, argmax_tokens



@triton.jit
def _softcap_sample_reduce_partial_kernel(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    temperature: torch.Tensor,
    partial_max: torch.Tensor,
    partial_denom: torch.Tensor,
    partial_expected: torch.Tensor,
    partial_greedy_value: torch.Tensor,
    partial_greedy_token: torch.Tensor,
    partial_sample_value: torch.Tensor,
    partial_sample_token: torch.Tensor,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    num_vocab_blocks: tl.constexpr,
    softcap: tl.constexpr,
    seed: tl.constexpr,
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

    temp = tl.load(temperature + row_offsets, mask=row_offsets < rows, other=1.0)
    temp_safe = tl.maximum(temp, 1.0e-10)
    z = acc / softcap
    scaled = libdevice.tanh(z) * softcap
    scaled = scaled / temp_safe[:, None]
    valid = (row_offsets[:, None] < rows) & (vocab_offsets[None, :] < vocab_size)
    scaled = tl.where(valid, scaled, -float("inf"))

    tile_max = tl.max(scaled, axis=1)
    weights = tl.exp(scaled - tile_max[:, None])
    weights = tl.where(valid, weights, 0.0)
    tile_denom = tl.sum(weights, axis=1)
    scaled_for_expected = tl.where(valid, scaled, 0.0)
    tile_expected = tl.sum(weights * scaled_for_expected, axis=1)

    greedy_candidates = tl.where(
        scaled == tile_max[:, None], vocab_offsets[None, :], vocab_size
    )
    tile_greedy_token = tl.min(greedy_candidates, axis=1)

    # Prototype token-index-stable RNG. This intentionally keeps each random
    # value keyed by absolute (row, token, seed), so chunk boundaries do not
    # affect sampled-token ids. A PR-ready version should use vLLM Philox.
    seed_u = seed % 65536
    row_u = row_offsets[:, None].to(tl.uint32) + 1 + seed_u
    tok_u = vocab_offsets[None, :].to(tl.uint32) + 1
    x = tok_u * 747796405 + row_u * 2891336453 + seed_u
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737
    x = (x >> 22) ^ x
    mantissa = x & 16777215
    uniform = (mantissa.to(tl.float32) + 0.5) * 5.960464477539063e-8
    gumbel = -tl.log(-tl.log(uniform))
    noisy = scaled + gumbel * (temp[:, None] > 0.0)
    noisy = tl.where(valid, noisy, -float("inf"))
    tile_sample_value = tl.max(noisy, axis=1)
    sample_candidates = tl.where(
        noisy == tile_sample_value[:, None], vocab_offsets[None, :], vocab_size
    )
    tile_sample_token = tl.min(sample_candidates, axis=1)

    out_offsets = row_offsets * num_vocab_blocks + vocab_block
    row_mask = row_offsets < rows
    tl.store(partial_max + out_offsets, tile_max, mask=row_mask)
    tl.store(partial_denom + out_offsets, tile_denom, mask=row_mask)
    tl.store(partial_expected + out_offsets, tile_expected, mask=row_mask)
    tl.store(partial_greedy_value + out_offsets, tile_max, mask=row_mask)
    tl.store(partial_greedy_token + out_offsets, tile_greedy_token, mask=row_mask)
    tl.store(partial_sample_value + out_offsets, tile_sample_value, mask=row_mask)
    tl.store(partial_sample_token + out_offsets, tile_sample_token, mask=row_mask)


@triton.jit
def _softcap_soft_embed_kernel(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    temperature: torch.Tensor,
    lse: torch.Tensor,
    soft_embed: torch.Tensor,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    embed_size: tl.constexpr,
    softcap: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    embed_block = tl.program_id(1)

    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    embed_offsets = embed_block * BLOCK_E + tl.arange(0, BLOCK_E)
    k_offsets = tl.arange(0, BLOCK_K)
    vocab_offsets = tl.arange(0, BLOCK_N)

    temp = tl.load(temperature + row_offsets, mask=row_offsets < rows, other=1.0)
    temp_safe = tl.maximum(temp, 1.0e-10)
    row_lse = tl.load(lse + row_offsets, mask=row_offsets < rows, other=0.0)
    out = tl.zeros((BLOCK_M, BLOCK_E), tl.float32)

    v_start = 0
    while v_start < vocab_size:
        v = v_start + vocab_offsets
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, hidden_size, BLOCK_K):
            k = k_start + k_offsets
            hidden_tile = tl.load(
                hidden + row_offsets[:, None] * hidden_size + k[None, :],
                mask=(row_offsets[:, None] < rows) & (k[None, :] < hidden_size),
                other=0.0,
            )
            weight_tile = tl.load(
                weight + v[:, None] * hidden_size + k[None, :],
                mask=(v[:, None] < vocab_size) & (k[None, :] < hidden_size),
                other=0.0,
            )
            acc += tl.dot(hidden_tile, tl.trans(weight_tile))

        z = acc / softcap
        scaled = libdevice.tanh(z) * softcap
        scaled = scaled / temp_safe[:, None]
        valid_vocab = v < vocab_size
        probs = tl.exp(scaled - row_lse[:, None])
        probs = tl.where((row_offsets[:, None] < rows) & valid_vocab[None, :],
                         probs, 0.0)
        embed_tile = tl.load(
            embed_weight + v[:, None] * embed_size + embed_offsets[None, :],
            mask=(v[:, None] < vocab_size)
            & (embed_offsets[None, :] < embed_size),
            other=0.0,
        )
        out += tl.dot(probs.to(tl.bfloat16), embed_tile)
        v_start += BLOCK_N

    tl.store(
        soft_embed + row_offsets[:, None] * embed_size + embed_offsets[None, :],
        out,
        mask=(row_offsets[:, None] < rows) & (embed_offsets[None, :] < embed_size),
    )

def diffusion_gemma_softcap_reductions_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    soft_embed_chunk_size: int = _DEFAULT_SOFT_EMBED_CHUNK,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute softcapped reductions plus chunked soft embeddings.

    The row reductions use the fused Triton helper. Soft embeddings use a
    chunked tensor-core bridge: recompute softcapped logits over vocab chunks,
    multiply probabilities and embedding weights in bf16, and accumulate chunks
    in fp32. This avoids full ``[rows, vocab]`` logits/probabilities while
    preserving the numerics needed for the later fully fused kernel.
    """
    lse, entropy, argmax_tokens = diffusion_gemma_softcap_lse_entropy_argmax(
        hidden,
        weight,
        softcap,
        temperature,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )

    if embed_weight.ndim != 2:
        raise ValueError("embed_weight must be a rank-2 tensor")
    if embed_weight.shape[0] != weight.shape[0]:
        raise ValueError("embed_weight vocab dimension must match weight")
    if not embed_weight.is_cuda:
        raise ValueError("embed_weight must be a CUDA tensor")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if soft_embed_chunk_size <= 0:
        raise ValueError("soft_embed_chunk_size must be positive")

    rows = hidden.shape[0]
    embed_size = embed_weight.shape[1]
    if rows == 0:
        soft_embeds = torch.empty(
            (0, embed_size), device=hidden.device, dtype=torch.float32
        )
        return lse, entropy, argmax_tokens, soft_embeds

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    vocab_size = weight.shape[0]
    soft_embeds = torch.zeros(
        (rows, embed_size), device=hidden.device, dtype=torch.float32
    )
    for start in range(0, vocab_size, soft_embed_chunk_size):
        end = min(start + soft_embed_chunk_size, vocab_size)
        logits = hidden @ weight[start:end].t()
        scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
        probs = torch.exp(scaled - lse[:, None])
        soft_embeds.add_(
            (probs.to(embed_weight.dtype) @ embed_weight[start:end]).float()
        )
    return lse, entropy, argmax_tokens, soft_embeds


def diffusion_gemma_softcap_shard_state(
    hidden: torch.Tensor,
    weight_shard: torch.Tensor,
    embed_weight_shard: torch.Tensor,
    softcap: float,
    temperature: float,
    *,
    vocab_start: int = 0,
    soft_embed_chunk_size: int = _DEFAULT_SOFT_EMBED_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Compute mergeable online state for one vocab shard.

    Returns row-wise ``(max, denom, expected, soft_embed, argmax_value,
    argmax_token)``. ``denom``, ``expected``, and ``soft_embed`` are
    unnormalized to the returned row max, so multiple shards can be merged with
    the same flash-style rescaling used across vocab chunks.
    """
    if hidden.ndim != 2 or weight_shard.ndim != 2 or embed_weight_shard.ndim != 2:
        raise ValueError("hidden, weight_shard, and embed_weight_shard must be rank-2")
    if hidden.shape[1] != weight_shard.shape[1]:
        raise ValueError("hidden and weight_shard hidden dimensions must match")
    if weight_shard.shape[0] != embed_weight_shard.shape[0]:
        raise ValueError("shard vocab dimensions must match")
    if not hidden.is_cuda or not weight_shard.is_cuda or not embed_weight_shard.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if embed_weight_shard.dtype != torch.bfloat16:
        raise ValueError("embed_weight_shard must be bfloat16 for this prototype")
    if weight_shard.shape[0] == 0:
        raise ValueError("weight_shard must have at least one vocab row")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if soft_embed_chunk_size <= 0:
        raise ValueError("soft_embed_chunk_size must be positive")

    hidden = hidden.contiguous()
    weight_shard = weight_shard.contiguous()
    embed_weight_shard = embed_weight_shard.contiguous()
    rows = hidden.shape[0]
    shard_vocab = weight_shard.shape[0]
    embed_size = embed_weight_shard.shape[1]
    device = hidden.device
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device,
                                 dtype=torch.float32)
        return empty, empty, empty, empty_soft, empty, empty_tokens

    running_max = torch.full((rows,), -torch.inf, device=device,
                             dtype=torch.float32)
    denom = torch.zeros((rows,), device=device, dtype=torch.float32)
    expected = torch.zeros((rows,), device=device, dtype=torch.float32)
    soft_embed = torch.zeros((rows, embed_size), device=device,
                             dtype=torch.float32)
    argmax_value = torch.full((rows,), -torch.inf, device=device,
                              dtype=torch.float32)
    argmax_token = torch.zeros((rows,), device=device, dtype=torch.int64)

    for start in range(0, shard_vocab, soft_embed_chunk_size):
        end = min(start + soft_embed_chunk_size, shard_vocab)
        logits = hidden @ weight_shard[start:end].t()
        scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
        tile_max = scaled.max(dim=-1).values
        weights = torch.exp(scaled - tile_max[:, None])
        tile_denom = weights.sum(dim=-1)
        tile_expected = (weights * scaled).sum(dim=-1)
        tile_soft_embed = (
            weights.to(embed_weight_shard.dtype)
            @ embed_weight_shard[start:end]
        ).float()

        new_max = torch.maximum(running_max, tile_max)
        old_scale = torch.exp(running_max - new_max)
        tile_scale = torch.exp(tile_max - new_max)
        denom = denom * old_scale + tile_denom * tile_scale
        expected = expected * old_scale + tile_expected * tile_scale
        soft_embed = (
            soft_embed * old_scale[:, None]
            + tile_soft_embed * tile_scale[:, None]
        )
        running_max = new_max

        tile_argmax_value = tile_max
        tile_argmax_local = scaled.argmax(dim=-1)
        tile_argmax_token = tile_argmax_local.to(torch.int64) + vocab_start + start
        update_argmax = tile_argmax_value > argmax_value
        argmax_value = torch.where(update_argmax, tile_argmax_value, argmax_value)
        argmax_token = torch.where(update_argmax, tile_argmax_token, argmax_token)

    return running_max, denom, expected, soft_embed, argmax_value, argmax_token


def diffusion_gemma_softcap_online_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float,
    *,
    soft_embed_chunk_size: int = _DEFAULT_SOFT_EMBED_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reductions and soft embeddings with one online vocab pass.

    This is the single-rank form of the vocab-shard state contract. It streams
    vocab chunks once, keeps online softmax state in fp32, and normalizes only
    after all chunks are merged.
    """
    row_max, denom, expected, soft_embed, _, argmax_token = (
        diffusion_gemma_softcap_shard_state(
            hidden,
            weight,
            embed_weight,
            softcap,
            temperature,
            vocab_start=0,
            soft_embed_chunk_size=soft_embed_chunk_size,
        )
    )
    if hidden.shape[0] == 0:
        return row_max, denom, argmax_token, soft_embed
    lse = row_max + denom.log()
    entropy = lse - expected / denom
    return lse, entropy, argmax_token, soft_embed / denom[:, None]


def _stable_uniform_from_indices(
    row_offsets: torch.Tensor,
    token_offsets: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Generate deterministic U(0, 1) values keyed by row and token id.

    This is a prototype stateless RNG for chunked Gumbel-max. Each random value
    depends only on ``(row, token, seed)``, so changing chunk size or vocab-shard
    boundaries does not change the sampled token. A serving implementation
    should replace this with vLLM's Philox/counter RNG contract.
    """
    x = (
        token_offsets[None, :].to(torch.int64)
        + (row_offsets[:, None].to(torch.int64) + 1) * _INT64_MIX_A
        + int(seed)
    )
    x = (x ^ (x >> 30)) * _INT64_MIX_B
    x = (x ^ (x >> 27)) * _INT64_MIX_C
    x = x ^ (x >> 31)
    mantissa = torch.bitwise_and(x, _INT64_MASK_53).to(torch.float64)
    return ((mantissa + 0.5) * _FLOAT_2_NEG_53).to(torch.float32)


def diffusion_gemma_softcap_gumbel_shard_state(
    hidden: torch.Tensor,
    weight_shard: torch.Tensor,
    softcap: float,
    temperature: float,
    seed: int,
    *,
    vocab_start: int = 0,
    chunk_size: int = _GUMBEL_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one vocab shard with token-index-stable Gumbel-max state."""
    if hidden.ndim != 2 or weight_shard.ndim != 2:
        raise ValueError("hidden and weight_shard must be rank-2 tensors")
    if hidden.shape[1] != weight_shard.shape[1]:
        raise ValueError("hidden and weight_shard hidden dimensions must match")
    if not hidden.is_cuda or not weight_shard.is_cuda:
        raise ValueError("hidden and weight_shard must be CUDA tensors")
    if weight_shard.shape[0] == 0:
        raise ValueError("weight_shard must have at least one vocab row")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    hidden = hidden.contiguous()
    weight_shard = weight_shard.contiguous()
    rows = hidden.shape[0]
    shard_vocab = weight_shard.shape[0]
    device = hidden.device
    if rows == 0:
        empty_values = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        return empty_values, empty_tokens

    row_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    best_value = torch.full((rows,), -torch.inf, device=device,
                            dtype=torch.float32)
    best_token = torch.zeros((rows,), device=device, dtype=torch.int64)

    for start in range(0, shard_vocab, chunk_size):
        end = min(start + chunk_size, shard_vocab)
        token_offsets = torch.arange(
            vocab_start + start, vocab_start + end, device=device,
            dtype=torch.int64
        )
        logits = hidden @ weight_shard[start:end].t()
        scaled = (
            torch.tanh(logits.float() / softcap)
            * softcap
            / max(float(temperature), 1e-10)
        )
        if temperature > 0:
            uniform = _stable_uniform_from_indices(row_offsets, token_offsets,
                                                   seed)
            gumbel = -torch.log(-torch.log(uniform))
            noisy = scaled + gumbel
        else:
            noisy = scaled
        tile_value, tile_local = noisy.max(dim=-1)
        tile_token = tile_local.to(torch.int64) + vocab_start + start
        update = tile_value > best_value
        best_value = torch.where(update, tile_value, best_value)
        best_token = torch.where(update, tile_token, best_token)

    return best_value, best_token


def diffusion_gemma_merge_gumbel_shard_states(
    shard_states: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Merge per-shard Gumbel-max states with lowest-token tie breaking."""
    if not shard_states:
        raise ValueError("at least one shard state is required")

    shard_values = torch.stack([state[0] for state in shard_states])
    shard_tokens = torch.stack([state[1] for state in shard_states])
    best_value = shard_values.max(dim=0).values
    max_int = torch.iinfo(shard_tokens.dtype).max
    candidate_tokens = torch.where(
        shard_values == best_value[None, :],
        shard_tokens,
        torch.full_like(shard_tokens, max_int),
    )
    return candidate_tokens.min(dim=0).values


def diffusion_gemma_softcap_gumbel_sample(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float,
    temperature: float,
    seed: int,
    *,
    chunk_size: int = _GUMBEL_CHUNK,
) -> torch.Tensor:
    """Sample from softmax(softcapped logits / temperature) without full noise."""
    state = diffusion_gemma_softcap_gumbel_shard_state(
        hidden,
        weight,
        softcap,
        temperature,
        seed,
        vocab_start=0,
        chunk_size=chunk_size,
    )
    return diffusion_gemma_merge_gumbel_shard_states([state])


def diffusion_gemma_softcap_online_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float | torch.Tensor,
    seed: int,
    *,
    soft_embed_chunk_size: int = _DEFAULT_SOFT_EMBED_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Compute sampler-shaped outputs in one token-index-stable vocab pass.

    Returns ``(lse, entropy, sampled_token, greedy_token, soft_embed)``. This is
    the single-rank prototype shape the serving sampler needs: sampled tokens
    for denoising, greedy tokens for convergence/history, entropy for
    acceptance, and soft embeddings for self-conditioning.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if isinstance(temperature, torch.Tensor):
        if temperature.ndim != 1 or temperature.shape[0] != hidden.shape[0]:
            raise ValueError("temperature tensor must have shape [rows]")
        if not temperature.is_cuda:
            raise ValueError("temperature tensor must be CUDA")
        if (temperature < 0).any():
            raise ValueError("temperature must be non-negative")
        temperature = temperature.to(device=hidden.device, dtype=torch.float32)
    elif temperature < 0:
        raise ValueError("temperature must be non-negative")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if soft_embed_chunk_size <= 0:
        raise ValueError("soft_embed_chunk_size must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    rows = hidden.shape[0]
    vocab = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device,
                                 dtype=torch.float32)
        return empty, empty, empty_tokens, empty_tokens, empty_soft

    row_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    running_max = torch.full((rows,), -torch.inf, device=device,
                             dtype=torch.float32)
    denom = torch.zeros((rows,), device=device, dtype=torch.float32)
    expected = torch.zeros((rows,), device=device, dtype=torch.float32)
    soft_embed = torch.zeros((rows, embed_size), device=device,
                             dtype=torch.float32)
    greedy_value = torch.full((rows,), -torch.inf, device=device,
                              dtype=torch.float32)
    greedy_token = torch.zeros((rows,), device=device, dtype=torch.int64)
    sample_value = torch.full((rows,), -torch.inf, device=device,
                              dtype=torch.float32)
    sample_token = torch.zeros((rows,), device=device, dtype=torch.int64)

    for start in range(0, vocab, soft_embed_chunk_size):
        end = min(start + soft_embed_chunk_size, vocab)
        token_offsets = torch.arange(start, end, device=device,
                                     dtype=torch.int64)
        logits = hidden @ weight[start:end].t()
        if isinstance(temperature, torch.Tensor):
            temp_safe = temperature[:, None].clamp(min=1e-10)
            noise_scale = (temperature[:, None] > 0).to(torch.float32)
        else:
            temp_safe = max(float(temperature), 1e-10)
            noise_scale = 1.0 if temperature > 0 else 0.0
        scaled = torch.tanh(logits.float() / softcap) * softcap / temp_safe

        tile_max = scaled.max(dim=-1).values
        weights = torch.exp(scaled - tile_max[:, None])
        tile_denom = weights.sum(dim=-1)
        tile_expected = (weights * scaled).sum(dim=-1)
        tile_soft_embed = (
            weights.to(embed_weight.dtype) @ embed_weight[start:end]
        ).float()

        new_max = torch.maximum(running_max, tile_max)
        old_scale = torch.exp(running_max - new_max)
        tile_scale = torch.exp(tile_max - new_max)
        denom = denom * old_scale + tile_denom * tile_scale
        expected = expected * old_scale + tile_expected * tile_scale
        soft_embed = (
            soft_embed * old_scale[:, None]
            + tile_soft_embed * tile_scale[:, None]
        )
        running_max = new_max

        tile_greedy_value = tile_max
        tile_greedy_local = scaled.argmax(dim=-1)
        tile_greedy_token = tile_greedy_local.to(torch.int64) + start
        update_greedy = tile_greedy_value > greedy_value
        greedy_value = torch.where(update_greedy, tile_greedy_value,
                                   greedy_value)
        greedy_token = torch.where(update_greedy, tile_greedy_token,
                                   greedy_token)

        uniform = _stable_uniform_from_indices(row_offsets, token_offsets, seed)
        noisy = scaled + (-torch.log(-torch.log(uniform))) * noise_scale
        tile_sample_value, tile_sample_local = noisy.max(dim=-1)
        tile_sample_token = tile_sample_local.to(torch.int64) + start
        update_sample = tile_sample_value > sample_value
        sample_value = torch.where(update_sample, tile_sample_value,
                                   sample_value)
        sample_token = torch.where(update_sample, tile_sample_token,
                                   sample_token)

    lse = running_max + denom.log()
    entropy = lse - expected / denom
    return lse, entropy, sample_token, greedy_token, soft_embed / denom[:, None]


def diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float | torch.Tensor,
    seed: int,
    *,
    chunk_size: int = 8192,
    row_seed_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Exact chunked-cuBLAS streamed sampler with bounded scratch.

    This is the council-recommended fallback between the memory-heavy
    materialized path and the monolithic Triton prototype. It uses optimized
    tensor-core GEMMs while bounding full-vocab intermediates to
    ``[rows, chunk_size]``:

    1. pass over vocab chunks to compute LSE/entropy/greedy/Gumbel sample;
    2. pass over vocab chunks again to compute ``softmax(scaled) @ embed``.

    The second pass intentionally recomputes logits to avoid keeping full
    ``[rows, vocab]`` logits/probs/noisy tensors alive. This trades an extra
    cuBLAS projection for predictable memory and avoids Triton HBM accumulator
    traffic.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if isinstance(temperature, torch.Tensor):
        if temperature.ndim != 1 or temperature.shape[0] != hidden.shape[0]:
            raise ValueError("temperature tensor must have shape [rows]")
        if not temperature.is_cuda:
            raise ValueError("temperature tensor must be CUDA")
        if (temperature < 0).any():
            raise ValueError("temperature must be non-negative")
        temperature = temperature.to(device=hidden.device, dtype=torch.float32)
    elif temperature < 0:
        raise ValueError("temperature must be non-negative")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    rows = hidden.shape[0]
    vocab = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device,
                                 dtype=torch.float32)
        return empty, empty, empty_tokens, empty_tokens, empty_soft

    if isinstance(temperature, torch.Tensor):
        temp_safe = temperature.clamp(min=1e-10)
        noise_scale = (temperature > 0).to(torch.float32)
    else:
        temp_safe = max(float(temperature), 1e-10)
        noise_scale = 1.0 if temperature > 0 else 0.0

    if row_seed_offsets is None:
        row_seed_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    else:
        if row_seed_offsets.ndim != 1 or row_seed_offsets.shape[0] != rows:
            raise ValueError("row_seed_offsets must have shape [rows]")
        if not row_seed_offsets.is_cuda:
            raise ValueError("row_seed_offsets must be CUDA")
        row_seed_offsets = row_seed_offsets.to(device=device, dtype=torch.int64)

    if isinstance(temperature, torch.Tensor):
        zero_temp_rows = temperature <= 0
    else:
        zero_temp_rows = torch.full((rows,), temperature <= 0, device=device,
                                    dtype=torch.bool)

    running_max = torch.full((rows,), -torch.inf, device=device,
                             dtype=torch.float32)
    denom = torch.zeros((rows,), device=device, dtype=torch.float32)
    expected = torch.zeros((rows,), device=device, dtype=torch.float32)
    greedy_value = torch.full((rows,), -torch.inf, device=device,
                              dtype=torch.float32)
    greedy_token = torch.zeros((rows,), device=device, dtype=torch.int64)
    sample_value = torch.full((rows,), -torch.inf, device=device,
                              dtype=torch.float32)
    sample_token = torch.zeros((rows,), device=device, dtype=torch.int64)

    # Pass 1: reductions and sampled/greedy token ids.
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        token_offsets = torch.arange(start, end, device=device,
                                     dtype=torch.int64)
        logits = hidden @ weight[start:end].t()
        unscaled = torch.tanh(logits.float() / softcap) * softcap
        if isinstance(temp_safe, torch.Tensor):
            scaled = unscaled / temp_safe[:, None]
            scaled = torch.where(zero_temp_rows[:, None], unscaled, scaled)
            chunk_noise_scale = noise_scale[:, None]
        else:
            scaled = unscaled / temp_safe
            if temperature <= 0:
                scaled = unscaled
            chunk_noise_scale = noise_scale

        tile_max = scaled.max(dim=-1).values
        weights = torch.exp(scaled - tile_max[:, None])
        tile_denom = weights.sum(dim=-1)
        tile_expected = (weights * scaled).sum(dim=-1)
        new_max = torch.maximum(running_max, tile_max)
        old_scale = torch.exp(running_max - new_max)
        tile_scale = torch.exp(tile_max - new_max)
        denom = denom * old_scale + tile_denom * tile_scale
        expected = expected * old_scale + tile_expected * tile_scale
        running_max = new_max

        tile_greedy_local = scaled.argmax(dim=-1)
        tile_greedy_token = tile_greedy_local.to(torch.int64) + start
        update_greedy = tile_max > greedy_value
        greedy_value = torch.where(update_greedy, tile_max, greedy_value)
        greedy_token = torch.where(update_greedy, tile_greedy_token,
                                   greedy_token)

        uniform = _stable_uniform_from_indices(row_seed_offsets, token_offsets, seed)
        uniform = uniform.clamp(
            min=torch.finfo(uniform.dtype).tiny,
            max=1.0 - torch.finfo(uniform.dtype).eps,
        )
        noisy = scaled + (-torch.log(-torch.log(uniform))) * chunk_noise_scale
        tile_sample_value, tile_sample_local = noisy.max(dim=-1)
        tile_sample_token = tile_sample_local.to(torch.int64) + start
        update_sample = tile_sample_value > sample_value
        sample_value = torch.where(update_sample, tile_sample_value,
                                   sample_value)
        sample_token = torch.where(update_sample, tile_sample_token,
                                   sample_token)

    lse = running_max + denom.log()
    entropy = lse - expected / denom
    entropy = torch.where(zero_temp_rows, torch.zeros_like(entropy), entropy)

    # Pass 2: soft embeddings from global LSE. Use bf16 matmul to match the
    # materialized sampler precision regime, but fp32 accumulation across chunks.
    soft_embed = torch.zeros((rows, embed_size), device=device,
                             dtype=torch.float32)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        logits = hidden @ weight[start:end].t()
        unscaled = torch.tanh(logits.float() / softcap) * softcap
        if isinstance(temp_safe, torch.Tensor):
            scaled = unscaled / temp_safe[:, None]
            scaled = torch.where(zero_temp_rows[:, None], unscaled, scaled)
        else:
            scaled = unscaled / temp_safe
            if temperature <= 0:
                scaled = unscaled
        probs = torch.exp(scaled - lse[:, None])
        soft_embed.add_(
            (probs.to(embed_weight.dtype) @ embed_weight[start:end]).float()
        )

    if zero_temp_rows.any():
        soft_embed[zero_temp_rows] = embed_weight[greedy_token[zero_temp_rows]].float()

    return lse, entropy, sample_token, greedy_token, soft_embed


def diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float | torch.Tensor,
    seed: int,
    *,
    row_chunk_size: int = 256,
    row_seed_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Exact row-chunked materialized streamed sampler.

    This is the simple bounded-memory baseline: keep the fast materialized
    PyTorch/cuBLAS math, but process decode rows in chunks so full-vocab
    transients are bounded by ``[row_chunk_size, vocab]`` instead of
    ``[rows, vocab]``. It is intentionally not as memory-frugal as the Triton
    prototypes, but it is much faster than vocab-chunked two-pass fallbacks and
    is a useful serving knob for high-pressure batches.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if isinstance(temperature, torch.Tensor):
        if temperature.ndim != 1 or temperature.shape[0] != hidden.shape[0]:
            raise ValueError("temperature tensor must have shape [rows]")
        if not temperature.is_cuda:
            raise ValueError("temperature tensor must be CUDA")
        if (temperature < 0).any():
            raise ValueError("temperature must be non-negative")
        temperature = temperature.to(device=hidden.device, dtype=torch.float32)
    elif temperature < 0:
        raise ValueError("temperature must be non-negative")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    rows = hidden.shape[0]
    vocab = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device,
                                 dtype=torch.float32)
        return empty, empty, empty_tokens, empty_tokens, empty_soft

    if row_seed_offsets is None:
        row_seed_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    else:
        if row_seed_offsets.ndim != 1 or row_seed_offsets.shape[0] != rows:
            raise ValueError("row_seed_offsets must have shape [rows]")
        if not row_seed_offsets.is_cuda:
            raise ValueError("row_seed_offsets must be CUDA")
        row_seed_offsets = row_seed_offsets.to(device=device, dtype=torch.int64)

    token_offsets = torch.arange(vocab, device=device, dtype=torch.int64)
    lse_out = torch.empty((rows,), device=device, dtype=torch.float32)
    entropy_out = torch.empty_like(lse_out)
    sample_out = torch.empty((rows,), device=device, dtype=torch.int64)
    greedy_out = torch.empty_like(sample_out)
    soft_out = torch.empty((rows, embed_size), device=device, dtype=torch.float32)

    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        h = hidden[row_start:row_end]
        logits = h @ weight.t()
        unscaled = torch.tanh(logits.float() / softcap) * softcap

        if isinstance(temperature, torch.Tensor):
            temp = temperature[row_start:row_end]
            zero_temp_rows = temp <= 0
            scaled = unscaled / temp.clamp(min=1e-10)[:, None]
            scaled = torch.where(zero_temp_rows[:, None], unscaled, scaled)
            noise_scale = (temp > 0).to(torch.float32)[:, None]
        else:
            zero_temp_rows = torch.full((row_end - row_start,),
                                        temperature <= 0,
                                        device=device,
                                        dtype=torch.bool)
            if temperature <= 0:
                scaled = unscaled
                noise_scale = 0.0
            else:
                scaled = unscaled / max(float(temperature), 1e-10)
                noise_scale = 1.0

        lse = scaled.logsumexp(dim=-1)
        probs = scaled.softmax(dim=-1)
        entropy = lse - (probs * scaled).sum(dim=-1)
        entropy = torch.where(zero_temp_rows, torch.zeros_like(entropy),
                              entropy)
        greedy = scaled.argmax(dim=-1)
        soft = (probs.to(embed_weight.dtype) @ embed_weight).float()

        uniform = _stable_uniform_from_indices(
            row_seed_offsets[row_start:row_end], token_offsets, seed)
        uniform = uniform.clamp(
            min=torch.finfo(uniform.dtype).tiny,
            max=1.0 - torch.finfo(uniform.dtype).eps,
        )
        noisy = scaled + (-torch.log(-torch.log(uniform))) * noise_scale
        sample = noisy.argmax(dim=-1)

        if zero_temp_rows.any():
            sample = torch.where(zero_temp_rows, greedy, sample)
            soft[zero_temp_rows] = embed_weight[greedy[zero_temp_rows]].float()

        lse_out[row_start:row_end] = lse
        entropy_out[row_start:row_end] = entropy
        sample_out[row_start:row_end] = sample.to(torch.int64)
        greedy_out[row_start:row_end] = greedy.to(torch.int64)
        soft_out[row_start:row_end] = soft

    return lse_out, entropy_out, sample_out, greedy_out, soft_out



def _merge_token_ties(
    values: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    best = values.max(dim=1).values
    max_int = torch.iinfo(tokens.dtype).max
    candidates = torch.where(
        values == best[:, None], tokens, torch.full_like(tokens, max_int)
    )
    return candidates.min(dim=1).values


def diffusion_gemma_softcap_triton_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
    seed: int,
    *,
    block_m: int = 16,
    block_n: int = 128,
    block_k: int = _DEFAULT_BLOCK_K,
    block_e: int = 64,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Triton full-output streaming sampler prototype.

    This is the real fused-backend shape, not the eager chunk-loop bridge: the
    heavy vocab scan runs inside Triton kernels and never materializes full
    ``[rows, vocab]`` logits/probs/noise. It uses two GPU phases: first row
    reductions/sample state, then soft-embedding accumulation from the computed
    LSE. The second phase intentionally recomputes logits per embedding tile;
    it is a correctness-first prototype for the full fused backend.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if temperature.ndim != 1 or temperature.shape[0] != hidden.shape[0]:
        raise ValueError("temperature tensor must have shape [rows]")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if not temperature.is_cuda:
        raise ValueError("temperature must be CUDA")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or block_e <= 0:
        raise ValueError("block sizes must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    temperature = temperature.to(device=hidden.device, dtype=torch.float32).contiguous()
    rows, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab_size == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device,
                                 dtype=torch.float32)
        return empty, empty, empty_tokens, empty_tokens, empty_soft

    num_vocab_blocks = triton.cdiv(vocab_size, block_n)
    partial_shape = (rows, num_vocab_blocks)
    partial_max = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_denom = torch.empty_like(partial_max)
    partial_expected = torch.empty_like(partial_max)
    partial_greedy_value = torch.empty_like(partial_max)
    partial_greedy_token = torch.empty(partial_shape, device=device,
                                       dtype=torch.int64)
    partial_sample_value = torch.empty_like(partial_max)
    partial_sample_token = torch.empty(partial_shape, device=device,
                                       dtype=torch.int64)
    grid_reduce = (triton.cdiv(rows, block_m), num_vocab_blocks)
    _softcap_sample_reduce_partial_kernel[grid_reduce](
        hidden,
        weight,
        temperature,
        partial_max,
        partial_denom,
        partial_expected,
        partial_greedy_value,
        partial_greedy_token,
        partial_sample_value,
        partial_sample_token,
        rows,
        hidden_size,
        vocab_size,
        num_vocab_blocks,
        float(softcap),
        int(seed),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )

    row_max = partial_max.max(dim=1).values
    rescale = torch.exp(partial_max - row_max[:, None])
    denom = (partial_denom * rescale).sum(dim=1)
    expected = (partial_expected * rescale).sum(dim=1) / denom
    lse = row_max + denom.log()
    entropy = lse - expected
    greedy_token = _merge_token_ties(partial_greedy_value, partial_greedy_token)
    sample_token = _merge_token_ties(partial_sample_value, partial_sample_token)

    soft_embed = torch.empty((rows, embed_size), device=device,
                             dtype=torch.float32)
    grid_soft = (triton.cdiv(rows, block_m), triton.cdiv(embed_size, block_e))
    _softcap_soft_embed_kernel[grid_soft](
        hidden,
        weight,
        embed_weight,
        temperature,
        lse,
        soft_embed,
        rows,
        hidden_size,
        vocab_size,
        embed_size,
        float(softcap),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_E=block_e,
        num_warps=num_warps,
    )
    return lse, entropy, sample_token, greedy_token, soft_embed


def diffusion_gemma_softcap_soft_embed_from_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    lse: torch.Tensor,
    softcap: float,
    temperature: float | torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 128,
    block_k: int = _DEFAULT_BLOCK_K,
    block_e: int = 64,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Compute only ``softmax(softcapped(hidden @ weight.T)) @ embed``.

    This is the pass-2 de-risk kernel for the fully fused DiffusionGemma
    sampler: callers provide the precomputed row-wise ``lse`` from pass 1, and
    this kernel recomputes score tiles while writing each output tile once.
    The ``lse`` contract is ``logsumexp(softcap(hidden @ weight.T) / temp)``;
    passing an un-temperatured normalizer silently computes the wrong embeds.
    It intentionally excludes Gumbel, entropy, TP merge, and serving wiring.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if lse.ndim != 1 or lse.shape[0] != hidden.shape[0]:
        raise ValueError("lse must have shape [rows]")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if not lse.is_cuda:
        raise ValueError("lse must be CUDA")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or block_e <= 0:
        raise ValueError("block sizes must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    lse = lse.to(device=hidden.device, dtype=torch.float32).contiguous()
    rows, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab_size == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        return torch.empty((0, embed_size), device=device, dtype=torch.float32)

    if isinstance(temperature, torch.Tensor):
        if temperature.ndim != 1 or temperature.shape[0] != rows:
            raise ValueError("temperature tensor must have shape [rows]")
        if not temperature.is_cuda:
            raise ValueError("temperature must be CUDA")
        if (temperature <= 0).any():
            raise ValueError("pass-2 softmax temperature must be positive")
        temperature = temperature.to(device=device, dtype=torch.float32).contiguous()
    else:
        if temperature <= 0:
            raise ValueError("pass-2 softmax temperature must be positive")
        temperature = torch.full((rows,), float(temperature), device=device,
                                 dtype=torch.float32)

    soft_embed = torch.empty((rows, embed_size), device=device,
                             dtype=torch.float32)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(embed_size, block_e))
    _softcap_soft_embed_kernel[grid](
        hidden,
        weight,
        embed_weight,
        temperature,
        lse,
        soft_embed,
        rows,
        hidden_size,
        vocab_size,
        embed_size,
        float(softcap),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_E=block_e,
        num_warps=num_warps,
    )
    return soft_embed


def diffusion_gemma_softcap_row_chunked_soft_embed_from_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    lse: torch.Tensor,
    softcap: float,
    temperature: float | torch.Tensor,
    *,
    row_chunk_size: int = 64,
) -> torch.Tensor:
    """Compute pass-2 soft embeddings with bounded row-chunk scratch.

    This is an isolated de-risk path between the full materialized reference and
    the extremely memory-frugal Triton recompute kernel.  It materializes one
    ``[row_chunk_size, vocab]`` score/probability scratch at a time, reuses that
    tile across all embedding columns through cuBLAS, and consumes the
    precomputed row-wise ``lse``.  It intentionally excludes Gumbel, entropy,
    TP merge, and serving wiring.

    The ``lse`` contract is ``logsumexp(softcap(hidden @ weight.T) / temp)``.
    Temperatures must be positive because zero-temperature rows are a greedy
    sampling special case rather than a softmax pass-2 contract.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if lse.ndim != 1 or lse.shape[0] != hidden.shape[0]:
        raise ValueError("lse must have shape [rows]")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if not lse.is_cuda:
        raise ValueError("lse must be CUDA")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    lse = lse.to(device=hidden.device, dtype=torch.float32).contiguous()
    rows = hidden.shape[0]
    vocab_size = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab_size == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        return torch.empty((0, embed_size), device=device, dtype=torch.float32)

    if isinstance(temperature, torch.Tensor):
        if temperature.ndim != 1 or temperature.shape[0] != rows:
            raise ValueError("temperature tensor must have shape [rows]")
        if not temperature.is_cuda:
            raise ValueError("temperature must be CUDA")
        if (temperature <= 0).any():
            raise ValueError("pass-2 softmax temperature must be positive")
        temperature = temperature.to(device=device, dtype=torch.float32).contiguous()
    else:
        if temperature <= 0:
            raise ValueError("pass-2 softmax temperature must be positive")
        temperature = torch.full((rows,), float(temperature), device=device,
                                 dtype=torch.float32)

    soft_embed = torch.empty((rows, embed_size), device=device,
                             dtype=torch.float32)
    weight_t = weight.t()
    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        # Match the Triton pass-2 prototype contract: bf16 tensor-core input
        # multiply with fp32 accumulated/output scores, then reconstruct probs
        # from the supplied LSE in-place to avoid stacked fp32 temporaries.
        scores = torch.mm(hidden[row_start:row_end], weight_t,
                          out_dtype=torch.float32)
        scores.div_(softcap).tanh_().mul_(softcap)
        scores.div_(temperature[row_start:row_end, None].clamp(min=1.0e-10))
        scores.sub_(lse[row_start:row_end, None]).exp_()
        probs_bf16 = scores.to(embed_weight.dtype)
        soft_embed[row_start:row_end] = torch.mm(
            probs_bf16, embed_weight, out_dtype=torch.float32
        )

    return soft_embed


@triton.jit
def _softcap_single_pass_sample_soft_embed_kernel(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    temperature: torch.Tensor,
    lse_out: torch.Tensor,
    entropy_out: torch.Tensor,
    sample_token_out: torch.Tensor,
    greedy_token_out: torch.Tensor,
    soft_embed: torch.Tensor,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    embed_size: tl.constexpr,
    softcap: tl.constexpr,
    seed: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
) -> None:
    row_block = tl.program_id(0)
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    vocab_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    embed_offsets_base = tl.arange(0, BLOCK_E)

    temp = tl.load(temperature + row_offsets, mask=row_offsets < rows, other=1.0)
    temp_safe = tl.maximum(temp, 1.0e-10)
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denom = tl.zeros((BLOCK_M,), tl.float32)
    expected = tl.zeros((BLOCK_M,), tl.float32)
    greedy_value = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    greedy_token = tl.zeros((BLOCK_M,), tl.int64)
    sample_value = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    sample_token = tl.zeros((BLOCK_M,), tl.int64)

    v_start = 0
    while v_start < vocab_size:
        v = v_start + vocab_offsets
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, hidden_size, BLOCK_K):
            k = k_start + k_offsets
            hidden_tile = tl.load(
                hidden + row_offsets[:, None] * hidden_size + k[None, :],
                mask=(row_offsets[:, None] < rows) & (k[None, :] < hidden_size),
                other=0.0,
            )
            weight_tile = tl.load(
                weight + v[:, None] * hidden_size + k[None, :],
                mask=(v[:, None] < vocab_size) & (k[None, :] < hidden_size),
                other=0.0,
            )
            acc += tl.dot(hidden_tile, tl.trans(weight_tile))

        z = acc / softcap
        scaled = libdevice.tanh(z) * softcap
        scaled = scaled / temp_safe[:, None]
        valid = (row_offsets[:, None] < rows) & (v[None, :] < vocab_size)
        scaled = tl.where(valid, scaled, -float("inf"))

        tile_max = tl.max(scaled, axis=1)
        # DiffusionGemma softcap bounds scaled logits to +/- softcap/temp.
        # Use that fixed row-wise shift to avoid rescaling the full
        # soft-embedding accumulator every vocab tile. This is exact up to fp32
        # underflow of negligible tail mass for positive temperatures.
        row_shift = softcap / temp_safe
        weights = tl.exp(scaled - row_shift[:, None])
        weights = tl.where(valid, weights, 0.0)
        tile_denom = tl.sum(weights, axis=1)
        tile_expected = tl.sum(weights * tl.where(valid, scaled, 0.0), axis=1)

        e_start = 0
        while e_start < embed_size:
            e = e_start + embed_offsets_base
            prev = tl.load(
                soft_embed + row_offsets[:, None] * embed_size + e[None, :],
                mask=(row_offsets[:, None] < rows) & (e[None, :] < embed_size),
                other=0.0,
            )
            prev = tl.where(v_start == 0, 0.0, prev)
            embed_tile = tl.load(
                embed_weight + v[:, None] * embed_size + e[None, :],
                mask=(v[:, None] < vocab_size) & (e[None, :] < embed_size),
                other=0.0,
            )
            tile_soft = tl.dot(weights.to(tl.bfloat16), embed_tile)
            updated = prev + tile_soft
            tl.store(
                soft_embed + row_offsets[:, None] * embed_size + e[None, :],
                updated,
                mask=(row_offsets[:, None] < rows) & (e[None, :] < embed_size),
            )
            e_start += BLOCK_E

        denom += tile_denom
        expected += tile_expected
        running_max = tl.maximum(running_max, tile_max)

        greedy_candidates = tl.where(
            scaled == tile_max[:, None], v[None, :], vocab_size
        )
        tile_greedy_token = tl.min(greedy_candidates, axis=1)
        update_greedy = tile_max > greedy_value
        greedy_value = tl.where(update_greedy, tile_max, greedy_value)
        greedy_token = tl.where(update_greedy, tile_greedy_token, greedy_token)

        seed_u = seed % 65536
        row_u = row_offsets[:, None].to(tl.uint32) + 1 + seed_u
        tok_u = v[None, :].to(tl.uint32) + 1
        x = tok_u * 747796405 + row_u * 2891336453 + seed_u
        x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737
        x = (x >> 22) ^ x
        mantissa = x & 16777215
        uniform = (mantissa.to(tl.float32) + 0.5) * 5.960464477539063e-8
        noisy = scaled + (-tl.log(-tl.log(uniform))) * (temp[:, None] > 0.0)
        noisy = tl.where(valid, noisy, -float("inf"))
        tile_sample_value = tl.max(noisy, axis=1)
        sample_candidates = tl.where(
            noisy == tile_sample_value[:, None], v[None, :], vocab_size
        )
        tile_sample_token = tl.min(sample_candidates, axis=1)
        update_sample = tile_sample_value > sample_value
        sample_value = tl.where(update_sample, tile_sample_value, sample_value)
        sample_token = tl.where(update_sample, tile_sample_token, sample_token)

        v_start += BLOCK_N

    final_shift = softcap / temp_safe
    lse = final_shift + tl.log(denom)
    entropy = lse - expected / denom
    e_start = 0
    while e_start < embed_size:
        e = e_start + embed_offsets_base
        val = tl.load(
            soft_embed + row_offsets[:, None] * embed_size + e[None, :],
            mask=(row_offsets[:, None] < rows) & (e[None, :] < embed_size),
            other=0.0,
        )
        tl.store(
            soft_embed + row_offsets[:, None] * embed_size + e[None, :],
            val / denom[:, None],
            mask=(row_offsets[:, None] < rows) & (e[None, :] < embed_size),
        )
        e_start += BLOCK_E

    row_mask = row_offsets < rows
    tl.store(lse_out + row_offsets, lse, mask=row_mask)
    tl.store(entropy_out + row_offsets, entropy, mask=row_mask)
    tl.store(sample_token_out + row_offsets, sample_token, mask=row_mask)
    tl.store(greedy_token_out + row_offsets, greedy_token, mask=row_mask)


def diffusion_gemma_softcap_triton_single_pass_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
    seed: int,
    *,
    block_m: int = 4,
    block_n: int = 128,
    block_k: int = _DEFAULT_BLOCK_K,
    block_e: int = 64,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Single-pass Triton full-output streamed sampler prototype.

    Unlike ``triton_sample_soft_embeds``, this scans vocab once per row block and
    updates the full soft-embedding accumulator in HBM as the online softmax max
    changes. It avoids both full logits materialization and per-embed-tile logit
    recomputation.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if temperature.ndim != 1 or temperature.shape[0] != hidden.shape[0]:
        raise ValueError("temperature tensor must have shape [rows]")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if not temperature.is_cuda:
        raise ValueError("temperature must be CUDA")
    if embed_weight.dtype != torch.bfloat16:
        raise ValueError("embed_weight must be bfloat16 for this prototype")
    if softcap <= 0:
        raise ValueError("softcap must be positive")

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    embed_weight = embed_weight.contiguous()
    temperature = temperature.to(device=hidden.device, dtype=torch.float32).contiguous()
    rows, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab_size == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device,
                                 dtype=torch.float32)
        return empty, empty, empty_tokens, empty_tokens, empty_soft

    lse = torch.empty((rows,), device=device, dtype=torch.float32)
    entropy = torch.empty_like(lse)
    sample = torch.empty((rows,), device=device, dtype=torch.int64)
    greedy = torch.empty_like(sample)
    soft_embed = torch.empty((rows, embed_size), device=device,
                             dtype=torch.float32)
    grid = (triton.cdiv(rows, block_m),)
    _softcap_single_pass_sample_soft_embed_kernel[grid](
        hidden,
        weight,
        embed_weight,
        temperature,
        lse,
        entropy,
        sample,
        greedy,
        soft_embed,
        rows,
        hidden_size,
        vocab_size,
        embed_size,
        float(softcap),
        int(seed),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_E=block_e,
        num_warps=num_warps,
    )
    return lse, entropy, sample, greedy, soft_embed

def diffusion_gemma_merge_softcap_shard_states(
    shard_states: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge vocab-shard states into LSE, entropy, argmax, and soft embeds."""
    if not shard_states:
        raise ValueError("at least one shard state is required")

    shard_max = torch.stack([state[0] for state in shard_states])
    shard_denom = torch.stack([state[1] for state in shard_states])
    shard_expected = torch.stack([state[2] for state in shard_states])
    shard_soft = torch.stack([state[3] for state in shard_states])
    shard_argmax_value = torch.stack([state[4] for state in shard_states])
    shard_argmax_token = torch.stack([state[5] for state in shard_states])

    row_max = shard_max.max(dim=0).values
    rescale = torch.exp(shard_max - row_max[None, :])
    denom = (shard_denom * rescale).sum(dim=0)
    expected = (shard_expected * rescale).sum(dim=0)
    soft_embed = (shard_soft * rescale[:, :, None]).sum(dim=0) / denom[:, None]
    lse = row_max + denom.log()
    entropy = lse - expected / denom

    argmax_value = shard_argmax_value.max(dim=0).values
    max_int = torch.iinfo(shard_argmax_token.dtype).max
    candidate_tokens = torch.where(
        shard_argmax_value == argmax_value[None, :],
        shard_argmax_token,
        torch.full_like(shard_argmax_token, max_int),
    )
    argmax_token = candidate_tokens.min(dim=0).values
    return lse, entropy, argmax_token, soft_embed


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
    """Compute row-wise softcapped lm-head logsumexp without full logits."""
    lse, _, _ = diffusion_gemma_softcap_lse_entropy_argmax(
        hidden,
        weight,
        softcap,
        temperature,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )
    return lse
