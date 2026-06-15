# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma_fused_lse import (
    diffusion_gemma_softcap_lse,
    diffusion_gemma_softcap_lse_entropy_argmax,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton LSE tests"
)


def _reference_softcap_reductions(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
    lse = scaled.logsumexp(dim=-1)
    probs = scaled.softmax(dim=-1)
    expected = (probs * scaled).sum(dim=-1)
    entropy = lse - expected
    return lse, entropy, scaled.argmax(dim=-1)


def _make_inputs(
    rows: int,
    hidden_size: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(rows + hidden_size + vocab_size)
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
    ) / hidden_size**0.5
    return hidden, weight


def _assert_selected_token_is_near_materialized_max(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    selected_tokens: torch.Tensor,
    softcap: float,
    temperature: float,
):
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
    selected = scaled.gather(1, selected_tokens[:, None]).squeeze(1)
    torch.testing.assert_close(selected, scaled.max(dim=-1).values, rtol=0,
                               atol=1e-3)


@pytest.mark.parametrize(
    ("rows", "hidden_size", "vocab_size", "block_n"),
    [
        (1, 64, 129, 128),
        (37, 130, 777, 128),
        (64, 256, 4099, 256),
    ],
)
def test_diffusion_gemma_softcap_lse_entropy_argmax_matches_reference(
    rows: int,
    hidden_size: int,
    vocab_size: int,
    block_n: int,
):
    hidden, weight = _make_inputs(rows, hidden_size, vocab_size)

    actual_lse, actual_entropy, actual_argmax = (
        diffusion_gemma_softcap_lse_entropy_argmax(
            hidden,
            weight,
            softcap=30.0,
            temperature=0.7,
            block_m=16,
            block_n=block_n,
            block_k=64,
            num_warps=4,
        )
    )
    expected_lse, expected_entropy, _ = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=0.7
    )

    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_entropy, expected_entropy, rtol=1e-3,
                               atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual_argmax, softcap=30.0, temperature=0.7
    )
    assert actual_lse.dtype == torch.float32
    assert actual_entropy.dtype == torch.float32
    assert actual_argmax.dtype == torch.int64


def test_diffusion_gemma_softcap_argmax_matches_unique_materialized_max():
    hidden = torch.eye(4, 16, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(32, 16, device="cuda", dtype=torch.bfloat16)
    expected_tokens = torch.tensor([3, 7, 11, 15], device="cuda")
    weight[expected_tokens, torch.arange(4, device="cuda")] = 8.0

    _, _, actual_tokens = diffusion_gemma_softcap_lse_entropy_argmax(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
        block_m=4,
        block_n=16,
        block_k=16,
        num_warps=4,
    )

    torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)


def test_diffusion_gemma_softcap_lse_wrapper_matches_full_helper():
    hidden, weight = _make_inputs(rows=17, hidden_size=96, vocab_size=513)

    actual = diffusion_gemma_softcap_lse(
        hidden,
        weight,
        softcap=30.0,
        temperature=1.3,
        block_m=16,
        block_n=128,
        block_k=64,
        num_warps=4,
    )
    expected, _, _ = diffusion_gemma_softcap_lse_entropy_argmax(
        hidden,
        weight,
        softcap=30.0,
        temperature=1.3,
        block_m=16,
        block_n=128,
        block_k=64,
        num_warps=4,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_diffusion_gemma_softcap_lse_entropy_argmax_handles_empty_rows():
    hidden = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((257, 64), device="cuda", dtype=torch.bfloat16)

    lse, entropy, argmax = diffusion_gemma_softcap_lse_entropy_argmax(
        hidden, weight, softcap=30.0, temperature=0.7
    )

    assert lse.shape == (0,)
    assert entropy.shape == (0,)
    assert argmax.shape == (0,)
    assert lse.dtype == torch.float32
    assert entropy.dtype == torch.float32
    assert argmax.dtype == torch.int64


def test_diffusion_gemma_softcap_lse_entropy_argmax_avoids_full_vocab_peak_memory():
    rows = 256
    hidden_size = 512
    vocab_size = 32768
    hidden, weight = _make_inputs(rows, hidden_size, vocab_size)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    expected = _reference_softcap_reductions(hidden, weight, softcap=30.0,
                                             temperature=0.7)
    torch.cuda.synchronize()
    full_peak = torch.cuda.max_memory_allocated() - base

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    actual = diffusion_gemma_softcap_lse_entropy_argmax(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
    )
    torch.cuda.synchronize()
    fused_peak = torch.cuda.max_memory_allocated() - base

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual[2], softcap=30.0, temperature=0.7
    )
    assert fused_peak < full_peak / 4
