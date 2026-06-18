# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    _get_diffusion_gemma_sampler_memory_reserve_bytes,
)


def test_diffusion_sampler_memory_reserve_disabled_by_default():
    assert (
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "",
            1.0,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=256,
            vocab_size=262144,
        )
        == 0
    )


def test_diffusion_sampler_memory_reserve_accepts_explicit_mib():
    assert (
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "512",
            1.0,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=256,
            vocab_size=262144,
        )
        == 512 * (1 << 20)
    )


def test_diffusion_sampler_memory_reserve_strips_whitespace():
    assert (
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            " auto ",
            1.0,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            canvas_length=256,
            vocab_size=1024,
        )
        == 256 * 1024 * 4 * 2
    )


def test_diffusion_sampler_memory_reserve_accepts_uppercase_auto():
    assert (
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            " AUTO ",
            1.0,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            canvas_length=256,
            vocab_size=1024,
        )
        == 256 * 1024 * 4 * 2
    )


def test_diffusion_sampler_memory_reserve_explicit_mib_ignores_scale():
    assert (
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "512",
            -1.0,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=256,
            vocab_size=262144,
        )
        == 512 * (1 << 20)
    )


def test_diffusion_sampler_memory_reserve_auto_estimate():
    reserve = _get_diffusion_gemma_sampler_memory_reserve_bytes(
        "auto",
        1.0,
        canvas_length=256,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        vocab_size=262144,
    )

    assert reserve == 16 * 256 * 262144 * 4 * 2


def test_diffusion_sampler_memory_reserve_auto_respects_token_cap():
    reserve = _get_diffusion_gemma_sampler_memory_reserve_bytes(
        "auto",
        1.25,
        canvas_length=256,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        vocab_size=1024,
    )

    # max_num_batched_tokens limits the materialized decode shape to 16
    # diffusion requests, not max_num_seqs=64.
    assert reserve == int(16 * 256 * 1024 * 4 * 2 * 1.25)


def test_diffusion_sampler_memory_reserve_rejects_negative():
    with pytest.raises(ValueError):
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "-1",
            1.0,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=256,
            vocab_size=262144,
        )


def test_diffusion_sampler_memory_reserve_rejects_invalid_value():
    with pytest.raises(ValueError, match="auto"):
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "not-a-number",
            1.0,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=256,
            vocab_size=262144,
        )


def test_diffusion_sampler_memory_reserve_rejects_negative_scale():
    with pytest.raises(ValueError):
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "auto",
            -0.1,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=256,
            vocab_size=262144,
        )


def test_diffusion_sampler_memory_reserve_rejects_invalid_canvas_length():
    with pytest.raises(ValueError, match="canvas_length"):
        _get_diffusion_gemma_sampler_memory_reserve_bytes(
            "auto",
            1.0,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            canvas_length=0,
            vocab_size=262144,
        )


def test_diffusion_model_state_reports_global_vocab_reserve(monkeypatch):
    state = DiffusionGemmaModelState.__new__(DiffusionGemmaModelState)
    state.max_num_reqs = 64
    state.max_num_tokens = 4096
    state.diffusion_states = SimpleNamespace(canvas_length=256)
    state.model_config = SimpleNamespace(get_vocab_size=lambda: 262144)

    monkeypatch.setattr(
        envs,
        "VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_MIB",
        "auto",
        raising=False,
    )
    monkeypatch.setattr(
        envs,
        "VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_SCALE",
        1.1,
        raising=False,
    )

    assert state.get_extra_non_kv_cache_memory_bytes() == int(
        16 * 256 * 262144 * 4 * 2 * 1.1
    )
