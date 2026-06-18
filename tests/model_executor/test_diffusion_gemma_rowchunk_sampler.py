# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    _get_diffusion_gemma_row_chunk_size,
    _is_diffusion_gemma_hidden_state_input,
    _is_diffusion_gemma_sampler_warmup_batch,
    _resolve_diffusion_gemma_sampler_backend_for_rows,
)
from vllm.model_executor.models.diffusion_gemma_rowchunk import (
    _uniform_from_mantissa,
    diffusion_gemma_softcap_row_chunked_sample_soft_embeds,
    stable_uniform_from_indices,
)


def test_diffusion_gemma_row_chunk_size_from_scratch_budget(monkeypatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK", raising=False)
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB", "1024")
    assert _get_diffusion_gemma_row_chunk_size(4096, 262144) == 64
    assert _get_diffusion_gemma_row_chunk_size(64, 262144) == 64

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK", "17")
    assert _get_diffusion_gemma_row_chunk_size(4096, 262144) == 17
    assert _get_diffusion_gemma_row_chunk_size(8, 262144) == 8


def test_diffusion_gemma_sampler_backend_auto_threshold(monkeypatch):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_AUTO_MAX_MATERIALIZED_ROWS", "2048")
    assert _resolve_diffusion_gemma_sampler_backend_for_rows("materialized", 9999) == (
        "materialized"
    )
    assert _resolve_diffusion_gemma_sampler_backend_for_rows("row_chunked", 1) == (
        "row_chunked"
    )
    assert _resolve_diffusion_gemma_sampler_backend_for_rows("auto", 2048) == (
        "materialized"
    )
    assert _resolve_diffusion_gemma_sampler_backend_for_rows("auto", 2049) == (
        "row_chunked"
    )



def test_diffusion_gemma_sampler_warmup_batch_detection():
    warmup = SimpleNamespace(num_reqs=2, req_ids=["_warmup_0_", "_warmup_1_"])
    mixed = SimpleNamespace(num_reqs=2, req_ids=["_warmup_0_", "real-req"])
    empty = SimpleNamespace(num_reqs=0, req_ids=[])

    assert _is_diffusion_gemma_sampler_warmup_batch(warmup)
    assert not _is_diffusion_gemma_sampler_warmup_batch(mixed)
    assert not _is_diffusion_gemma_sampler_warmup_batch(empty)


def test_diffusion_gemma_hidden_state_input_classification():
    hidden = torch.empty((3, 64))
    logits = torch.empty((3, 257))
    bad = torch.empty((3, 128))

    assert _is_diffusion_gemma_hidden_state_input(hidden, 257, 64)
    assert not _is_diffusion_gemma_hidden_state_input(logits, 257, 64)
    with pytest.raises(ValueError, match="expected hidden states"):
        _is_diffusion_gemma_hidden_state_input(bad, 257, 64)


def _legacy_stable_uniform_from_indices(
    row_offsets: torch.Tensor, token_offsets: torch.Tensor, seed: int
) -> torch.Tensor:
    x = (
        token_offsets[None, :].to(torch.int64)
        + (row_offsets[:, None].to(torch.int64) + 1) * -7046029254386353131
        + int(seed)
    )
    x = (x ^ (x >> 30)) * -4658895280553007687
    x = (x ^ (x >> 27)) * -7723592293110705685
    x = x ^ (x >> 31)
    mantissa = torch.bitwise_and(x, (1 << 53) - 1).to(torch.float64)
    return ((mantissa + 0.5) * (1.0 / float(1 << 53))).to(torch.float32)


def _legacy_uniform_from_mantissa(mantissa: torch.Tensor) -> torch.Tensor:
    return ((mantissa.to(torch.float64) + 0.5) * (1.0 / float(1 << 53))).to(
        torch.float32
    )


def _make_inputs(rows: int, hidden_size: int, vocab_size: int):
    generator = torch.Generator(device="cuda").manual_seed(2026061801)
    hidden = torch.randn(
        rows,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        vocab_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return hidden, weight


def _make_embed_weight(vocab_size: int, embed_size: int):
    generator = torch.Generator(device="cuda").manual_seed(2026061802)
    return torch.randn(
        vocab_size,
        embed_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )


def _reference_sample_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
    seed: int,
    row_seed_offsets: torch.Tensor,
):
    logits = hidden @ weight.t()
    unscaled = torch.tanh(logits.float() / softcap) * softcap
    zero_temp_rows = temperature <= 0
    scaled = unscaled / temperature.clamp(min=1e-10)[:, None]
    scaled = torch.where(zero_temp_rows[:, None], unscaled, scaled)
    lse = scaled.logsumexp(dim=-1)
    probs = scaled.softmax(dim=-1)
    entropy = lse - (probs * scaled).sum(dim=-1)
    entropy = torch.where(zero_temp_rows, torch.zeros_like(entropy), entropy)
    greedy = scaled.argmax(dim=-1)
    soft = (probs.to(embed_weight.dtype) @ embed_weight).float()
    token_offsets = torch.arange(weight.shape[0], device="cuda", dtype=torch.int64)
    uniform = stable_uniform_from_indices(row_seed_offsets, token_offsets, seed)
    uniform = uniform.clamp(
        min=torch.finfo(uniform.dtype).tiny,
        max=1.0 - torch.finfo(uniform.dtype).eps,
    )
    noisy = scaled + (-torch.log(-torch.log(uniform))) * (
        temperature > 0
    ).float()[:, None]
    sample = noisy.argmax(dim=-1)
    if zero_temp_rows.any():
        sample = torch.where(zero_temp_rows, greedy, sample)
        soft[zero_temp_rows] = embed_weight[greedy[zero_temp_rows]].float()
    return lse, entropy, sample, greedy, soft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stable_uniform_matches_legacy_float64_stream(monkeypatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK_FAST_FP32_RNG", raising=False)
    rows = torch.tensor([0, 1_000_003, 17_000_051], device="cuda", dtype=torch.int64)
    tokens = torch.arange(1024, device="cuda", dtype=torch.int64)

    actual = stable_uniform_from_indices(rows, tokens, seed=2026061807)
    expected = _legacy_stable_uniform_from_indices(rows, tokens, seed=2026061807)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_fp32_uniform_is_explicitly_not_bit_exact():
    mantissas = torch.tensor(
        [
            0,
            1,
            2,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 51) + (1 << 27),
            (1 << 52) - 1,
            1 << 52,
            (1 << 52) + 1,
            (1 << 53) - 3,
            (1 << 53) - 2,
            (1 << 53) - 1,
        ],
        device="cuda",
        dtype=torch.int64,
    )

    legacy = _uniform_from_mantissa(mantissas, fast_fp32=False)
    expected = _legacy_uniform_from_mantissa(mantissas)
    fast = _uniform_from_mantissa(mantissas, fast_fp32=True)

    torch.testing.assert_close(legacy, expected, rtol=0, atol=0)
    assert torch.any(legacy != fast)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stable_uniform_fast_fp32_env_is_opt_in(monkeypatch):
    rows = torch.tensor([0, 1_000_003, 17_000_051], device="cuda", dtype=torch.int64)
    tokens = torch.arange(2048, device="cuda", dtype=torch.int64)

    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK_FAST_FP32_RNG", raising=False)
    legacy = stable_uniform_from_indices(rows, tokens, seed=2026061818)

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK_FAST_FP32_RNG", "1")
    fast = stable_uniform_from_indices(rows, tokens, seed=2026061818)

    # The fast path is deterministic/statistically compatible, but not promised
    # to preserve every fp32 midpoint of the legacy stream. Natural hash samples
    # may or may not hit an adversarial mantissa, so this test only verifies the
    # opt-in path shape/range here; adversarial divergence is tested above.
    assert fast.shape == legacy.shape
    assert fast.dtype == torch.float32
    assert torch.all(fast > 0)
    assert torch.all(fast <= 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stable_uniform_from_indices_distribution_moments(monkeypatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_ROW_CHUNK_FAST_FP32_RNG", raising=False)
    rows = torch.arange(16, device="cuda", dtype=torch.int64) * 1_000_003
    tokens = torch.arange(8192, device="cuda", dtype=torch.int32)

    uniform = stable_uniform_from_indices(rows, tokens, seed=2026061806)

    assert uniform.dtype == torch.float32
    assert torch.all(uniform > 0)
    assert torch.all(uniform < 1)
    assert abs(uniform.mean().item() - 0.5) < 2e-3
    assert abs(uniform.var(unbiased=False).item() - (1.0 / 12.0)) < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_matches_materialized_reference():
    hidden, weight = _make_inputs(rows=9, hidden_size=96, vocab_size=677)
    embed_weight = _make_embed_weight(vocab_size=677, embed_size=48)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 1.3, 0.9, 0.8, 0.0, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.tensor(
        [8, 1, 7, 3, 5, 11, 13, 17, 19], device="cuda", dtype=torch.int64
    )

    expected = _reference_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061803,
        row_seed_offsets=row_seed_offsets,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061803,
        row_chunk_size=4,
        row_seed_offsets=row_seed_offsets,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_positive_temperature_fast_path_matches():
    hidden, weight = _make_inputs(rows=6, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.9, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.arange(6, device="cuda", dtype=torch.int64) * 5

    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061819,
        row_chunk_size=3,
        row_seed_offsets=row_seed_offsets,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061819,
        row_chunk_size=3,
        row_seed_offsets=row_seed_offsets,
        temperature_is_positive=True,
    )

    for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if idx in (2, 3):
            torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
        elif idx == 4:
            torch.testing.assert_close(
                actual_item, expected_item, rtol=2e-2, atol=2e-2
            )
        else:
            torch.testing.assert_close(actual_item, expected_item, rtol=3e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_chunk_size_invariant():
    hidden, weight = _make_inputs(rows=7, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.0, 0.9, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.arange(7, device="cuda", dtype=torch.int64) * 3
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061804,
        row_chunk_size=7,
        row_seed_offsets=row_seed_offsets,
    )

    for row_chunk_size in (1, 2, 3):
        actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=2026061804,
            row_chunk_size=row_chunk_size,
            row_seed_offsets=row_seed_offsets,
        )
        for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            if idx in (2, 3):
                torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
            elif idx == 4:
                torch.testing.assert_close(
                    actual_item, expected_item, rtol=2e-2, atol=2e-2
                )
            else:
                torch.testing.assert_close(
                    actual_item, expected_item, rtol=3e-4, atol=2e-3
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diffusion_gemma_row_chunked_row_seed_offsets_are_reorder_stable():
    hidden, weight = _make_inputs(rows=8, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 0.6, 0.9, 0.8, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.tensor(
        [11, 3, 20, 7, 15, 99, 2, 42], device="cuda", dtype=torch.int64
    )
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=2026061805,
        row_chunk_size=3,
        row_seed_offsets=row_seed_offsets,
    )
    perm = torch.tensor([3, 0, 7, 1, 6, 4, 2, 5], device="cuda")
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden[perm],
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature[perm],
        seed=2026061805,
        row_chunk_size=2,
        row_seed_offsets=row_seed_offsets[perm],
    )
    inv = torch.argsort(perm)
    for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if idx in (2, 3):
            torch.testing.assert_close(actual_item[inv], expected_item, rtol=0, atol=0)
        elif idx == 4:
            torch.testing.assert_close(
                actual_item[inv], expected_item, rtol=2e-2, atol=2e-2
            )
        else:
            torch.testing.assert_close(
                actual_item[inv], expected_item, rtol=3e-4, atol=2e-3
            )
