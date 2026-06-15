# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma_fused_lse import (
    diffusion_gemma_softcap_lse,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton LSE tests"
)


def _reference_softcap_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float,
    temperature: float,
) -> torch.Tensor:
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
    return scaled.logsumexp(dim=-1)


@pytest.mark.parametrize(
    ("rows", "hidden_size", "vocab_size"),
    [
        (1, 64, 129),
        (37, 130, 777),
        (64, 256, 4099),
    ],
)
def test_diffusion_gemma_softcap_lse_matches_materialized_reference(
    rows: int,
    hidden_size: int,
    vocab_size: int,
):
    torch.manual_seed(rows + hidden_size + vocab_size)
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
    ) / hidden_size**0.5

    actual = diffusion_gemma_softcap_lse(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
        block_m=16,
        block_n=128,
        block_k=64,
        num_warps=4,
    )
    expected = _reference_softcap_lse(hidden, weight, softcap=30.0,
                                      temperature=0.7)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert actual.dtype == torch.float32


def test_diffusion_gemma_softcap_lse_avoids_full_vocab_peak_memory():
    torch.manual_seed(2026)
    rows = 256
    hidden_size = 512
    vocab_size = 32768
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
    ) / hidden_size**0.5

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    expected = _reference_softcap_lse(hidden, weight, softcap=30.0,
                                      temperature=0.7)
    torch.cuda.synchronize()
    full_peak = torch.cuda.max_memory_allocated() - base

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    actual = diffusion_gemma_softcap_lse(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
    )
    torch.cuda.synchronize()
    fused_peak = torch.cuda.max_memory_allocated() - base

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert fused_peak < full_peak / 4
