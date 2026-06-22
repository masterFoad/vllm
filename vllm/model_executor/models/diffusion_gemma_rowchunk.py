# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-chunked DiffusionGemma sampler helpers.

This module intentionally keeps the production-candidate pressure path small:
it uses PyTorch/cuBLAS GEMMs over bounded row chunks and does not include the
experimental Triton or vocab-recompute prototypes.
"""

from __future__ import annotations

import torch

_INT64_MIX_A = -7046029254386353131
_INT64_MIX_B = -4658895280553007687
_INT64_MIX_C = -7723592293110705685
_INT64_MASK_53 = (1 << 53) - 1
_FLOAT_2_NEG_53 = 1.0 / float(1 << 53)


def _uniform_from_mantissa(mantissa: torch.Tensor) -> torch.Tensor:
    return ((mantissa.to(torch.float64) + 0.5) * _FLOAT_2_NEG_53).to(torch.float32)


def stable_uniform_from_indices(
    row_offsets: torch.Tensor,
    token_offsets: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Generate deterministic U(0, 1) values keyed by row and token id.

    Each value depends only on ``(row_offsets[i], token_offsets[j], seed)``.
    Changing row chunk size therefore does not change Gumbel-max samples.  This
    is a stateless pressure-mode RNG; a lower-level kernel can replace it with
    vLLM's Philox/counter RNG contract later.

    This preserves the original int64 hash / 53-bit mantissa / fp64-scale /
    fp32 stream exactly.
    """
    x = (
        token_offsets[None, :].to(torch.int64)
        + (row_offsets[:, None].to(torch.int64) + 1) * _INT64_MIX_A
        + int(seed)
    )
    x = (x ^ (x >> 30)) * _INT64_MIX_B
    x = (x ^ (x >> 27)) * _INT64_MIX_C
    x = x ^ (x >> 31)
    mantissa = torch.bitwise_and(x, _INT64_MASK_53)
    return _uniform_from_mantissa(mantissa)


def diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float | torch.Tensor,
    seed: int,
    *,
    row_chunk_size: int = 128,
    row_seed_offsets: torch.Tensor | None = None,
    token_offsets: torch.Tensor | None = None,
    temperature_is_positive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-chunked DiffusionGemma sampler using bounded materialized tiles.

    The fast materialized sampler builds full ``[rows, vocab]`` transients. This
    helper keeps the same dense cuBLAS-friendly math but bounds transients to
    ``[row_chunk_size, vocab]`` so high-concurrency decode can avoid OOM. The
    stateless Gumbel RNG is deterministic and chunk-size invariant, but it is
    not sample-stream identical to the default materialized sampler RNG.
    ``token_offsets`` is an optional precomputed ``torch.arange(vocab)`` for
    callers that reuse the helper every decode step.
    ``temperature_is_positive`` is a caller-provided precondition for tensor
    temperatures: set it only when every row temperature is strictly positive.
    It skips zero-temperature greedy handling to avoid a per-chunk CUDA sync.

    Returns ``(lse, entropy, sampled_tokens, greedy_tokens, soft_embeds)`` with
    one row per input hidden row. ``lse``, ``entropy``, ``greedy_tokens``, and
    ``soft_embeds`` describe the pre-Gumbel distribution; only
    ``sampled_tokens`` uses Gumbel noise.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or embed_weight.ndim != 2:
        raise ValueError("hidden, weight, and embed_weight must be rank-2")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight hidden dimensions must match")
    if weight.shape[0] != embed_weight.shape[0]:
        raise ValueError("vocab dimensions must match")
    if not hidden.is_cuda or not weight.is_cuda or not embed_weight.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if softcap <= 0:
        raise ValueError("softcap must be positive")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")

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

    rows = hidden.shape[0]
    vocab = weight.shape[0]
    embed_size = embed_weight.shape[1]
    device = hidden.device
    if vocab == 0:
        raise ValueError("weight must have at least one vocab row")
    if rows == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        empty_tokens = torch.empty((0,), device=device, dtype=torch.int64)
        empty_soft = torch.empty((0, embed_size), device=device, dtype=torch.float32)
        return empty, empty, empty_tokens, empty_tokens, empty_soft

    if row_seed_offsets is None:
        row_seed_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    else:
        if row_seed_offsets.ndim != 1 or row_seed_offsets.shape[0] != rows:
            raise ValueError("row_seed_offsets must have shape [rows]")
        if not row_seed_offsets.is_cuda:
            raise ValueError("row_seed_offsets must be CUDA")
        row_seed_offsets = row_seed_offsets.to(device=device, dtype=torch.int64)

    if token_offsets is None:
        token_offsets = torch.arange(vocab, device=device, dtype=torch.int64)
    else:
        if token_offsets.ndim != 1 or token_offsets.shape[0] != vocab:
            raise ValueError("token_offsets must have shape [vocab]")
        if token_offsets.device != device:
            raise ValueError("token_offsets must be on the same CUDA device")
        if token_offsets.dtype != torch.int64:
            raise ValueError("token_offsets must be int64")
    lse_out = torch.empty((rows,), device=device, dtype=torch.float32)
    entropy_out = torch.empty_like(lse_out)
    sample_out = torch.empty((rows,), device=device, dtype=torch.int64)
    greedy_out = torch.empty_like(sample_out)
    soft_out = torch.empty((rows, embed_size), device=device, dtype=torch.float32)

    # Avoid copying persistent model weights. Chunk temporaries are newly
    # allocated below, so in-place softcap/temperature transforms are safe.
    hidden = hidden.contiguous()

    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        logits = hidden[row_start:row_end] @ weight.t()
        scaled = logits.float()
        scaled.div_(float(softcap)).tanh_().mul_(float(softcap))

        if isinstance(temperature, torch.Tensor):
            temp = temperature[row_start:row_end]
            if temperature_is_positive:
                scaled = scaled / temp[:, None]
                zero_temp_rows = None
                has_zero_temp = False
                noise_scale = 1.0
            else:
                zero_temp_rows = temp <= 0
                scaled = scaled / temp.clamp(min=1e-10)[:, None]
                # For temp==0 DiffusionGemma uses greedy behavior. Keep finite
                # untemperatured softcapped scores for argmax/LSE diagnostics.
                has_zero_temp = bool(zero_temp_rows.any())
                if has_zero_temp:
                    unscaled = logits.float()
                    unscaled.div_(float(softcap)).tanh_().mul_(float(softcap))
                    scaled = torch.where(zero_temp_rows[:, None], unscaled, scaled)
                noise_scale = (temp > 0).to(torch.float32)[:, None]
            skip_gumbel = False
        else:
            scalar_zero_temp = float(temperature) <= 0
            zero_temp_rows = torch.full(
                (row_end - row_start,),
                scalar_zero_temp,
                device=device,
                dtype=torch.bool,
            )
            has_zero_temp = scalar_zero_temp
            skip_gumbel = scalar_zero_temp
            if not scalar_zero_temp:
                scaled.div_(float(temperature))
                noise_scale = 1.0
            else:
                noise_scale = 0.0

        greedy = scaled.argmax(dim=-1)
        log_probs = scaled.log_softmax(dim=-1)
        greedy_col = greedy[:, None]
        lse = (
            scaled.gather(1, greedy_col) - log_probs.gather(1, greedy_col)
        ).squeeze(1)
        probs = log_probs.exp_()
        entropy = lse - (probs * scaled).sum(dim=-1)
        if has_zero_temp:
            assert zero_temp_rows is not None
            entropy = torch.where(zero_temp_rows, torch.zeros_like(entropy), entropy)
        soft = (probs.to(embed_weight.dtype) @ embed_weight).float()

        if skip_gumbel:
            sample = greedy
        else:
            uniform = stable_uniform_from_indices(
                row_seed_offsets[row_start:row_end], token_offsets, seed
            )
            uniform.clamp_(
                min=torch.finfo(uniform.dtype).tiny,
                max=1.0 - torch.finfo(uniform.dtype).eps,
            )
            # In-place Gumbel transform on the throwaway uniform tile:
            # uniform = -log(-log(uniform)).
            uniform.log_()
            uniform.neg_()
            uniform.log_()
            uniform.neg_()
            if isinstance(noise_scale, torch.Tensor):
                uniform.mul_(noise_scale)
            # ``scaled`` is a chunk-local temporary. LSE, entropy, greedy, and
            # soft embeddings have already been computed, so reuse it for the
            # noisy argmax instead of allocating another full-vocab tile.
            scaled.add_(uniform)
            sample = scaled.argmax(dim=-1)

        if has_zero_temp:
            assert zero_temp_rows is not None
            sample = torch.where(zero_temp_rows, greedy, sample)
            soft[zero_temp_rows] = embed_weight[greedy[zero_temp_rows]].float()

        lse_out[row_start:row_end] = lse
        entropy_out[row_start:row_end] = entropy
        sample_out[row_start:row_end] = sample.to(torch.int64)
        greedy_out[row_start:row_end] = greedy.to(torch.int64)
        soft_out[row_start:row_end] = soft

    return lse_out, entropy_out, sample_out, greedy_out, soft_out
