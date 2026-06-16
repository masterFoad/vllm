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

    # Triton 3.6 in the current vLLM image does not expose tl.tanh. Use the
    # equivalent logistic form: tanh(z) = 2 / (1 + exp(-2z)) - 1.
    z = acc / softcap
    scaled = ((2.0 / (1.0 + tl.exp(-2.0 * z))) - 1.0) * softcap
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
    temperature: float,
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
    if temperature < 0:
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
        scaled = (
            torch.tanh(logits.float() / softcap)
            * softcap
            / max(float(temperature), 1e-10)
        )

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

        if temperature > 0:
            uniform = _stable_uniform_from_indices(row_offsets, token_offsets,
                                                   seed)
            noisy = scaled + (-torch.log(-torch.log(uniform)))
        else:
            noisy = scaled
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
