# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.diffusion_gemma import (
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
        == 256 * 1024 * 4
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

    assert reserve == 16 * 256 * 262144 * 4


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
    assert reserve == int(16 * 256 * 1024 * 4 * 1.25)


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
