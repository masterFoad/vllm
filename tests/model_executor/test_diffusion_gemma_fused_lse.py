# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models import diffusion_gemma
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    DiffusionSampler,
    _get_diffusion_gemma_streamed_backend,
    _get_diffusion_gemma_streamed_gumbel_settings,
    _get_diffusion_gemma_streamed_row_chunk_size,
    _resolve_diffusion_gemma_streamed_backend_for_rows,
)
from vllm.model_executor.models.diffusion_gemma_fused_lse import (
    _stable_uniform_from_indices,
    diffusion_gemma_resolve_gumbel_chunk_size,
    diffusion_gemma_merge_gumbel_shard_states,
    diffusion_gemma_merge_softcap_shard_states,
    diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds,
    diffusion_gemma_softcap_gumbel_sample,
    diffusion_gemma_softcap_gumbel_shard_state,
    diffusion_gemma_softcap_lse,
    diffusion_gemma_softcap_lse_entropy_argmax,
    diffusion_gemma_softcap_online_sample_soft_embeds,
    diffusion_gemma_softcap_online_soft_embeds,
    diffusion_gemma_softcap_reductions_soft_embeds,
    diffusion_gemma_softcap_row_chunked_sample_soft_embeds,
    diffusion_gemma_softcap_row_chunked_soft_embed_from_lse,
    diffusion_gemma_softcap_shard_state,
    diffusion_gemma_softcap_soft_embed_from_lse,
    diffusion_gemma_softcap_triton_sample_soft_embeds,
    diffusion_gemma_softcap_triton_single_pass_sample_soft_embeds,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton LSE tests"
)


class _FakeSamplingStates:

    def __init__(self, max_num_logprobs: int = -1):
        self._max_num_logprobs = max_num_logprobs

    def max_num_logprobs(self, _slots):
        return self._max_num_logprobs


class _FakeUvaBackedTensor:

    def __init__(self, size: int, dtype: torch.dtype):
        np_dtype = np.int64 if dtype is torch.int64 else np.int32
        self.np = np.zeros((size,), dtype=np_dtype)
        self.gpu = torch.zeros((size,), dtype=dtype, device="cuda")

    def copy_to_uva(self):
        self.gpu.copy_(torch.as_tensor(self.np, device="cuda"))


def test_diffusion_gemma_streamed_backend_validation(monkeypatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND", raising=False)
    monkeypatch.delenv(
        "VLLM_DIFFUSION_GEMMA_ALLOW_EXPERIMENTAL_TRITON", raising=False
    )
    assert _get_diffusion_gemma_streamed_backend() == "eager"

    for backend in ("auto", "eager", "cublas_two_pass", "row_chunked"):
        monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND", backend)
        assert _get_diffusion_gemma_streamed_backend() == backend

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND", "triton_full")
    monkeypatch.delenv(
        "VLLM_DIFFUSION_GEMMA_ALLOW_EXPERIMENTAL_TRITON", raising=False
    )
    with pytest.raises(ValueError, match="experimental"):
        _get_diffusion_gemma_streamed_backend()

    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_ALLOW_EXPERIMENTAL_TRITON", "1"
    )
    assert _get_diffusion_gemma_streamed_backend() == "triton_full"

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unsupported"):
        _get_diffusion_gemma_streamed_backend()


def test_diffusion_gemma_streamed_auto_backend_boundary(monkeypatch):
    monkeypatch.delenv(
        "VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS",
        raising=False,
    )
    assert (
        _resolve_diffusion_gemma_streamed_backend_for_rows("auto", 2048)
        == "materialized"
    )
    assert (
        _resolve_diffusion_gemma_streamed_backend_for_rows("auto", 2049)
        == "row_chunked"
    )
    assert (
        _resolve_diffusion_gemma_streamed_backend_for_rows("row_chunked", 1)
        == "row_chunked"
    )

    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS", "3072"
    )
    assert (
        _resolve_diffusion_gemma_streamed_backend_for_rows("auto", 3072)
        == "materialized"
    )
    assert (
        _resolve_diffusion_gemma_streamed_backend_for_rows("auto", 3073)
        == "row_chunked"
    )



def test_diffusion_gemma_streamed_row_chunk_size_from_scratch_budget(monkeypatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK", raising=False)
    monkeypatch.delenv(
        "VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK_SCRATCH_MIB",
        raising=False,
    )
    # Default 512 MiB budget gives 256 rows for a 262k vocab because
    # floor(512MiB / (262144 * 6B)) rounds down to a 128-multiple.
    assert _get_diffusion_gemma_streamed_row_chunk_size(2048, 262144) == 256
    assert _get_diffusion_gemma_streamed_row_chunk_size(64, 262144) == 64

    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK_SCRATCH_MIB", "256"
    )
    assert _get_diffusion_gemma_streamed_row_chunk_size(2048, 262144) == 128

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK", "384")
    assert _get_diffusion_gemma_streamed_row_chunk_size(2048, 262144) == 384
    assert _get_diffusion_gemma_streamed_row_chunk_size(128, 262144) == 128


def test_diffusion_gemma_streamed_gumbel_chunk_size_from_scratch_budget(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_STREAMED_GUMBEL_CHUNK",
                       raising=False)
    monkeypatch.delenv(
        "VLLM_DIFFUSION_GEMMA_STREAMED_GUMBEL_SCRATCH_MIB",
        raising=False,
    )

    # Default 1024 MiB budget gives the previous safe c16 value:
    # floor(1024MiB / (4096 rows * 32B conservative scratch/elem)) = 8192.
    chunk, scratch_mib, source = _get_diffusion_gemma_streamed_gumbel_settings(
        rows=4096, soft_embed_chunk_size=32768, vocab_size=262144)
    assert (chunk, scratch_mib, source) == (8192, 1024, "auto")

    # Lower row pressure can use a larger RNG tile without shrinking the GEMM
    # chunk; high row pressure shrinks only Gumbel/noisy-argmax scratch.
    assert diffusion_gemma_resolve_gumbel_chunk_size(
        rows=1024,
        soft_embed_chunk_size=32768,
        vocab_size=262144,
        requested_chunk=0,
        scratch_mib=1024,
    ) == 32768
    assert diffusion_gemma_resolve_gumbel_chunk_size(
        rows=8192,
        soft_embed_chunk_size=32768,
        vocab_size=262144,
        requested_chunk=0,
        scratch_mib=1024,
    ) == 4096

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_STREAMED_GUMBEL_CHUNK", "1234")
    chunk, scratch_mib, source = _get_diffusion_gemma_streamed_gumbel_settings(
        rows=4096, soft_embed_chunk_size=32768, vocab_size=262144)
    assert (chunk, scratch_mib, source) == (1234, 1024, "explicit:1234")

    # Explicit values and huge budgets never exceed the outer vocab tile.
    assert diffusion_gemma_resolve_gumbel_chunk_size(
        rows=1,
        soft_embed_chunk_size=181,
        vocab_size=541,
        requested_chunk=999999,
        scratch_mib=1024,
    ) == 181


def test_diffusion_gemma_sampler_uses_streamed_auto_row_chunked_path(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND", "auto")
    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS", "3"
    )
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK", raising=False)
    monkeypatch.setattr(diffusion_gemma, "UvaBackedTensor", _FakeUvaBackedTensor)
    monkeypatch.setattr(
        diffusion_gemma,
        "async_copy_to_gpu",
        lambda x, device=None, out=None: torch.as_tensor(x, device=device),
    )
    monkeypatch.setattr(
        diffusion_gemma,
        "_compute_num_rejected",
        lambda num_logits, num_sampled, query_start_loc: num_logits - num_sampled,
    )

    calls: list[dict[str, object]] = []

    def fake_row_chunked(
        hidden,
        lm_head_weight,
        embed_weight,
        softcap,
        temperature,
        seed,
        *,
        row_chunk_size,
        row_seed_offsets,
    ):
        calls.append(
            {
                "hidden_shape": tuple(hidden.shape),
                "row_chunk_size": row_chunk_size,
                "row_seed_offsets": row_seed_offsets.detach().cpu().tolist(),
                "seed": seed,
            }
        )
        rows = hidden.shape[0]
        device = hidden.device
        new_tokens = torch.arange(rows, device=device, dtype=torch.long) % 7
        entropy = torch.zeros(rows, device=device, dtype=torch.float32)
        soft_embeds = torch.ones(
            rows, embed_weight.shape[1], device=device, dtype=torch.float32
        )
        return (
            torch.zeros(rows, device=device, dtype=torch.float32),
            entropy,
            new_tokens,
            new_tokens.clone(),
            soft_embeds,
        )

    monkeypatch.setattr(
        diffusion_gemma,
        "diffusion_gemma_softcap_row_chunked_sample_soft_embeds",
        fake_row_chunked,
    )

    states = DiffusionGemmaRequestStates(
        max_num_reqs=2,
        canvas_length=2,
        vocab_size=7,
        max_denoising_steps=4,
        device=torch.device("cuda"),
        hidden_size=3,
        stability_threshold=2,
    )
    sampler = SimpleNamespace(
        sampling_states=_FakeSamplingStates(max_num_logprobs=-1),
        req_states=SimpleNamespace(
            draft_tokens=torch.zeros(2, 2, dtype=torch.long, device="cuda")
        ),
    )
    diffusion_sampler = DiffusionSampler(
        sampler=sampler,
        diffusion_config=SimpleNamespace(canvas_length=2),
        vocab_size=7,
        diffusion_states=states,
        confidence_threshold=1.0,
        t_min=1.0,
        t_max=1.0,
        entropy_bound=10.0,
        embed_weight=torch.randn(7, 3, device="cuda", dtype=torch.bfloat16),
        lm_head_weight=torch.randn(7, 3, device="cuda", dtype=torch.bfloat16),
        final_logit_softcapping=30.0,
        normalizer=torch.tensor(1.0, device="cuda"),
        use_streamed_sampler=True,
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=1,
        idx_mapping_np=np.array([0, 1], dtype=np.int64),
        idx_mapping=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
        cu_num_logits_np=np.array([0, 2, 4], dtype=np.int64),
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int64),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.long, device="cuda"),
    )

    output = diffusion_sampler(
        torch.randn(4, 3, device="cuda", dtype=torch.bfloat16),
        input_batch,
    )

    assert output.logprobs_tensors is None
    assert calls == [
        {
            "hidden_shape": (4, 3),
            "row_chunk_size": 4,
            "row_seed_offsets": [0, 1, 2, 3],
            "seed": 0,
        }
    ]
    torch.testing.assert_close(
        states.canvas[:2],
        torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.long),
        rtol=0,
        atol=0,
    )
    assert diffusion_sampler._stream_seed == 1


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


def _make_embed_weight(
    vocab_size: int,
    embed_size: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        vocab_size, embed_size, device="cuda", dtype=torch.bfloat16
    ) / embed_size**0.5


def _reference_soft_embeds(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float,
) -> torch.Tensor:
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
    probs = scaled.softmax(dim=-1)
    return (probs.to(embed_weight.dtype) @ embed_weight).float()


def _reference_soft_embeds_fp32(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: float,
) -> torch.Tensor:
    logits = hidden.float() @ weight.float().t()
    scaled = torch.tanh(logits / softcap) * softcap / temperature
    probs = scaled.softmax(dim=-1)
    return probs @ embed_weight.float()


def _reference_gumbel_sample(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    softcap: float,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / softcap) * softcap / temperature
    row_offsets = torch.arange(hidden.shape[0], device=hidden.device,
                               dtype=torch.int64)
    token_offsets = torch.arange(weight.shape[0], device=hidden.device,
                                 dtype=torch.int64)
    uniform = _stable_uniform_from_indices(row_offsets, token_offsets, seed)
    gumbel = -torch.log(-torch.log(uniform))
    return (scaled + gumbel).argmax(dim=-1)


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


def test_diffusion_gemma_softcap_reductions_soft_embeds_matches_reference():
    hidden, weight = _make_inputs(rows=19, hidden_size=128, vocab_size=769)
    embed_weight = _make_embed_weight(vocab_size=769, embed_size=96,
                                      seed=20260616)

    actual = diffusion_gemma_softcap_reductions_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        block_m=16,
        block_n=128,
        block_k=64,
        soft_embed_chunk_size=256,
        num_warps=4,
    )
    expected_lse, expected_entropy, _ = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=0.7
    )
    expected_soft_embeds = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=1e-3,
                               atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual[2], softcap=30.0, temperature=0.7
    )
    torch.testing.assert_close(actual[3], expected_soft_embeds, rtol=2e-2,
                               atol=2e-2)
    assert actual[3].dtype == torch.float32


def test_diffusion_gemma_softcap_reductions_soft_embeds_handles_empty_rows():
    hidden = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((257, 64), device="cuda", dtype=torch.bfloat16)
    embed_weight = torch.empty((257, 48), device="cuda", dtype=torch.bfloat16)

    lse, entropy, argmax, soft_embeds = (
        diffusion_gemma_softcap_reductions_soft_embeds(
            hidden, weight, embed_weight, softcap=30.0, temperature=0.7
        )
    )

    assert lse.shape == (0,)
    assert entropy.shape == (0,)
    assert argmax.shape == (0,)
    assert soft_embeds.shape == (0, 48)
    assert soft_embeds.dtype == torch.float32


def test_diffusion_gemma_softcap_reductions_soft_embeds_memory():
    rows = 128
    hidden_size = 256
    vocab_size = 16384
    embed_size = 256
    hidden, weight = _make_inputs(rows, hidden_size, vocab_size)
    embed_weight = _make_embed_weight(vocab_size, embed_size, seed=20260617)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    expected = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )
    torch.cuda.synchronize()
    full_peak = torch.cuda.max_memory_allocated() - base

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    actual = diffusion_gemma_softcap_reductions_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        soft_embed_chunk_size=1024,
    )
    torch.cuda.synchronize()
    fused_peak = torch.cuda.max_memory_allocated() - base

    torch.testing.assert_close(actual[3], expected, rtol=2e-2, atol=2e-2)
    assert fused_peak < full_peak / 2


def _shard_states(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    num_shards: int,
    softcap: float,
    temperature: float,
    chunk_size: int,
):
    shard_size = (weight.shape[0] + num_shards - 1) // num_shards
    states = []
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, weight.shape[0])
        if start == end:
            continue
        states.append(
            diffusion_gemma_softcap_shard_state(
                hidden,
                weight[start:end],
                embed_weight[start:end],
                softcap,
                temperature,
                vocab_start=start,
                soft_embed_chunk_size=chunk_size,
            )
        )
    return states


def test_diffusion_gemma_softcap_online_soft_embeds_matches_reference():
    hidden, weight = _make_inputs(rows=23, hidden_size=128, vocab_size=1543)
    embed_weight = _make_embed_weight(vocab_size=1543, embed_size=96,
                                      seed=20260619)

    actual = diffusion_gemma_softcap_online_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        soft_embed_chunk_size=256,
    )
    expected_lse, expected_entropy, _ = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=0.7
    )
    expected_soft_embeds = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=1e-3,
                               atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual[2], softcap=30.0, temperature=0.7
    )
    torch.testing.assert_close(actual[3], expected_soft_embeds, rtol=2e-2,
                               atol=2e-2)


def test_diffusion_gemma_softcap_online_soft_embeds_matches_peaked_fp32_reference():
    torch.manual_seed(20260620)
    hidden = torch.randn(13, 128, device="cuda", dtype=torch.bfloat16) * 2
    weight = torch.randn(997, 128, device="cuda", dtype=torch.bfloat16)
    embed_weight = _make_embed_weight(vocab_size=997, embed_size=64,
                                      seed=20260621)
    # Force a few high-confidence rows so this does not only cover the benign
    # near-uniform synthetic distribution.
    weight[:13] = hidden / hidden.float().norm(dim=-1, keepdim=True).to(
        hidden.dtype
    ) * 8

    actual = diffusion_gemma_softcap_online_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=5.0,
        temperature=0.3,
        soft_embed_chunk_size=127,
    )
    expected_soft_embeds = _reference_soft_embeds_fp32(
        hidden, weight, embed_weight, softcap=5.0, temperature=0.3
    )

    torch.testing.assert_close(actual[3], expected_soft_embeds, rtol=5e-2,
                               atol=5e-2)


def test_diffusion_gemma_softcap_online_soft_embeds_chunk_size_invariant():
    hidden, weight = _make_inputs(rows=11, hidden_size=96, vocab_size=677)
    embed_weight = _make_embed_weight(vocab_size=677, embed_size=48,
                                      seed=20260622)
    expected = diffusion_gemma_softcap_online_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        soft_embed_chunk_size=677,
    )

    for chunk_size in (1, 113, 256):
        actual = diffusion_gemma_softcap_online_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=0.7,
            soft_embed_chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-4,
                                   atol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-3,
                                   atol=1e-3)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=2e-2,
                                   atol=2e-2)


def test_diffusion_gemma_softcap_online_matches_triton_reductions():
    hidden, weight = _make_inputs(rows=17, hidden_size=128, vocab_size=1031)
    embed_weight = _make_embed_weight(vocab_size=1031, embed_size=32,
                                      seed=20260623)

    online = diffusion_gemma_softcap_online_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        soft_embed_chunk_size=257,
    )
    triton_lse, triton_entropy, triton_argmax = (
        diffusion_gemma_softcap_lse_entropy_argmax(
            hidden,
            weight,
            softcap=30.0,
            temperature=0.7,
            block_m=16,
            block_n=128,
            block_k=64,
            num_warps=4,
        )
    )

    torch.testing.assert_close(online[0], triton_lse, rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(online[1], triton_entropy, rtol=1e-3,
                               atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, triton_argmax, softcap=30.0, temperature=0.7
    )
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, online[2], softcap=30.0, temperature=0.7
    )


def test_diffusion_gemma_softcap_online_soft_embeds_handles_empty_rows():
    hidden = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((257, 64), device="cuda", dtype=torch.bfloat16)
    embed_weight = torch.empty((257, 48), device="cuda", dtype=torch.bfloat16)

    lse, entropy, argmax, soft_embeds = diffusion_gemma_softcap_online_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )

    assert lse.shape == (0,)
    assert entropy.shape == (0,)
    assert argmax.shape == (0,)
    assert soft_embeds.shape == (0, 48)
    assert soft_embeds.dtype == torch.float32


def test_diffusion_gemma_tp_shard_merge_matches_materialized_reference():
    hidden, weight = _make_inputs(rows=21, hidden_size=128, vocab_size=1009)
    embed_weight = _make_embed_weight(vocab_size=1009, embed_size=96,
                                      seed=20260618)

    states = _shard_states(
        hidden,
        weight,
        embed_weight,
        num_shards=4,
        softcap=30.0,
        temperature=0.7,
        chunk_size=128,
    )
    actual = diffusion_gemma_merge_softcap_shard_states(states)
    expected_lse, expected_entropy, _ = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=0.7
    )
    expected_soft_embeds = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=1e-3,
                               atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual[2], softcap=30.0, temperature=0.7
    )
    torch.testing.assert_close(actual[3], expected_soft_embeds, rtol=2e-2,
                               atol=2e-2)


@pytest.mark.parametrize(
    ("softcap", "temperature"),
    [
        (5.0, 0.1),
        (30.0, 1.0),
        (50.0, 5.0),
    ],
)
def test_diffusion_gemma_tp_shard_merge_handles_softcap_temperature_sweep(
    softcap: float,
    temperature: float,
):
    hidden, weight = _make_inputs(rows=9, hidden_size=96, vocab_size=521)
    embed_weight = _make_embed_weight(vocab_size=521, embed_size=64,
                                      seed=int(softcap * 100 + temperature * 10))

    states = _shard_states(
        hidden,
        weight,
        embed_weight,
        num_shards=3,
        softcap=softcap,
        temperature=temperature,
        chunk_size=96,
    )
    actual = diffusion_gemma_merge_softcap_shard_states(states)
    expected_lse, expected_entropy, _ = _reference_softcap_reductions(
        hidden, weight, softcap=softcap, temperature=temperature
    )
    expected_soft_embeds = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=softcap,
        temperature=temperature
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=5e-4, atol=3e-3)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=2e-3,
                               atol=2e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual[2], softcap=softcap, temperature=temperature
    )
    torch.testing.assert_close(actual[3], expected_soft_embeds, rtol=3e-2,
                               atol=3e-2)


def test_diffusion_gemma_tp_shard_merge_tie_breaks_to_lowest_token():
    hidden = torch.zeros((1, 16), device="cuda", dtype=torch.bfloat16)
    hidden[0, 0] = 1.0
    weight = torch.zeros((32, 16), device="cuda", dtype=torch.bfloat16)
    weight[2, 0] = 8.0
    weight[20, 0] = 8.0
    embed_weight = torch.zeros((32, 8), device="cuda", dtype=torch.bfloat16)

    states = _shard_states(
        hidden,
        weight,
        embed_weight,
        num_shards=4,
        softcap=30.0,
        temperature=0.7,
        chunk_size=8,
    )
    _, _, argmax_token, _ = diffusion_gemma_merge_softcap_shard_states(states)

    torch.testing.assert_close(
        argmax_token, torch.tensor([2], device="cuda"), rtol=0, atol=0
    )


def test_diffusion_gemma_chunked_gumbel_matches_materialized_reference():
    hidden, weight = _make_inputs(rows=9, hidden_size=96, vocab_size=677)
    expected = _reference_gumbel_sample(
        hidden, weight, softcap=30.0, temperature=0.7, seed=20260624
    )

    for chunk_size in (1, 113, 256, 677):
        actual = diffusion_gemma_softcap_gumbel_sample(
            hidden,
            weight,
            softcap=30.0,
            temperature=0.7,
            seed=20260624,
            chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_diffusion_gemma_chunked_gumbel_is_shard_layout_stable():
    hidden, weight = _make_inputs(rows=7, hidden_size=80, vocab_size=541)
    expected = diffusion_gemma_softcap_gumbel_sample(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
        seed=20260625,
        chunk_size=97,
    )

    for num_shards in (1, 3, 8):
        shard_size = (weight.shape[0] + num_shards - 1) // num_shards
        states = []
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, weight.shape[0])
            if start == end:
                continue
            states.append(
                diffusion_gemma_softcap_gumbel_shard_state(
                    hidden,
                    weight[start:end],
                    softcap=30.0,
                    temperature=0.7,
                    seed=20260625,
                    vocab_start=start,
                    chunk_size=53,
                )
            )
        actual = diffusion_gemma_merge_gumbel_shard_states(states)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_diffusion_gemma_chunked_gumbel_handles_empty_rows():
    hidden = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((257, 64), device="cuda", dtype=torch.bfloat16)

    actual = diffusion_gemma_softcap_gumbel_sample(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
        seed=20260626,
    )

    assert actual.shape == (0,)
    assert actual.dtype == torch.int64


def test_diffusion_gemma_chunked_gumbel_temperature_zero_is_greedy():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    expected = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=1.0
    )[2]

    actual = diffusion_gemma_softcap_gumbel_sample(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.0,
        seed=20260629,
        chunk_size=31,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_diffusion_gemma_chunked_gumbel_matches_softmax_distribution():
    rows = 8192
    scores = torch.tensor(
        [-2.0, -1.0, -0.3, 0.0, 0.4, 0.8, 1.2, 2.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden = torch.ones((rows, 1), device="cuda", dtype=torch.bfloat16)
    weight = scores[:, None].contiguous()

    samples = diffusion_gemma_softcap_gumbel_sample(
        hidden,
        weight,
        softcap=30.0,
        temperature=1.0,
        seed=20260628,
        chunk_size=3,
    )

    freq = torch.bincount(samples, minlength=scores.numel()).float() / rows
    scaled = torch.tanh(scores.float() / 30.0) * 30.0
    expected = scaled.softmax(dim=-1)
    total_variation = 0.5 * (freq - expected).abs().sum()
    assert total_variation.item() < 0.03


def test_diffusion_gemma_online_sample_soft_embeds_matches_component_helpers():
    hidden, weight = _make_inputs(rows=13, hidden_size=96, vocab_size=733)
    embed_weight = _make_embed_weight(vocab_size=733, embed_size=48,
                                      seed=20260630)

    actual = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        seed=20260631,
        soft_embed_chunk_size=97,
    )
    expected_lse, expected_entropy, expected_argmax, expected_soft = (
        diffusion_gemma_softcap_online_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=0.7,
            soft_embed_chunk_size=97,
        )
    )
    expected_sample = diffusion_gemma_softcap_gumbel_sample(
        hidden,
        weight,
        softcap=30.0,
        temperature=0.7,
        seed=20260631,
        chunk_size=97,
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected_sample, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected_argmax, rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected_soft, rtol=0, atol=0)


def test_diffusion_gemma_online_sample_soft_embeds_chunk_size_invariant():
    hidden, weight = _make_inputs(rows=7, hidden_size=80, vocab_size=541)
    embed_weight = _make_embed_weight(vocab_size=541, embed_size=32,
                                      seed=20260632)
    expected = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        seed=20260633,
        soft_embed_chunk_size=541,
    )

    for chunk_size in (1, 53, 128):
        actual = diffusion_gemma_softcap_online_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=0.7,
            seed=20260633,
            soft_embed_chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-4,
                                   atol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-3,
                                   atol=1e-3)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=2e-2,
                                   atol=2e-2)


def test_diffusion_gemma_online_sample_soft_embeds_gumbel_chunk_invariant():
    hidden, weight = _make_inputs(rows=6, hidden_size=80, vocab_size=541)
    embed_weight = _make_embed_weight(vocab_size=541, embed_size=32,
                                      seed=2026063201)
    expected = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        seed=2026063301,
        soft_embed_chunk_size=181,
        gumbel_chunk_size=181,
    )

    for gumbel_chunk_size in (None, -1, 0, 1, 7, 53, 128, 30000, 999999):
        actual = diffusion_gemma_softcap_online_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=0.7,
            seed=2026063301,
            soft_embed_chunk_size=181,
            gumbel_chunk_size=gumbel_chunk_size,
            gumbel_scratch_mib=1,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=0, atol=0)


def test_diffusion_gemma_online_sample_soft_embeds_row_seed_offsets_are_reorder_stable():
    hidden, weight = _make_inputs(rows=8, hidden_size=80, vocab_size=541)
    embed_weight = _make_embed_weight(vocab_size=541, embed_size=32,
                                      seed=2026063202)
    row_seed_offsets = torch.tensor([11, 3, 20, 7, 15, 99, 2, 42],
                                    device="cuda", dtype=torch.int64)
    temperature = torch.full((8,), 0.7, device="cuda", dtype=torch.float32)
    expected = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=temperature,
        seed=2026063302, soft_embed_chunk_size=113, gumbel_chunk_size=17,
        row_seed_offsets=row_seed_offsets,
    )

    perm = torch.tensor([5, 0, 7, 1, 6, 2, 4, 3], device="cuda")
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device="cuda")
    actual = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden[perm], weight, embed_weight, softcap=30.0,
        temperature=temperature[perm], seed=2026063302,
        soft_embed_chunk_size=113, gumbel_chunk_size=17,
        row_seed_offsets=row_seed_offsets[perm],
    )

    for idx in (0, 1, 2, 3, 4):
        tol = {0: (3e-4, 2e-3), 1: (1e-3, 1e-3), 4: (2e-2, 2e-2)}.get(idx, (0, 0))
        torch.testing.assert_close(actual[idx][inv], expected[idx],
                                   rtol=tol[0], atol=tol[1])


def test_diffusion_gemma_online_sample_soft_embeds_zero_temp_rows_are_greedy():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32,
                                      seed=2026063203)
    temperature = torch.tensor([0.0, 0.7, 0.0, 1.1, 0.0], device="cuda",
                               dtype=torch.float32)

    actual = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=temperature,
        seed=2026063303, soft_embed_chunk_size=64, gumbel_chunk_size=17,
    )

    zero_rows = temperature == 0
    torch.testing.assert_close(actual[2][zero_rows], actual[3][zero_rows],
                               rtol=0, atol=0)
    torch.testing.assert_close(actual[1][zero_rows],
                               torch.zeros_like(actual[1][zero_rows]),
                               rtol=0, atol=0)
    torch.testing.assert_close(
        actual[4][zero_rows],
        embed_weight[actual[3][zero_rows]].float(),
        rtol=0,
        atol=0,
    )


def test_diffusion_gemma_online_sample_soft_embeds_accepts_row_temperature():
    hidden, weight = _make_inputs(rows=6, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32,
                                      seed=20260634)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 1.3, 0.0], device="cuda", dtype=torch.float32
    )

    actual = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260635,
        soft_embed_chunk_size=63,
    )

    for row, temp in enumerate(temperature.tolist()):
        expected = diffusion_gemma_softcap_online_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temp,
            seed=20260635,
            soft_embed_chunk_size=63,
        )
        for idx in (0, 1, 2, 3, 4):
            torch.testing.assert_close(
                actual[idx][row : row + 1],
                expected[idx][row : row + 1],
                rtol=2e-2 if idx == 4 else 1e-3,
                atol=2e-2 if idx == 4 else 2e-3,
            )




def test_diffusion_gemma_cublas_two_pass_sample_soft_embeds_matches_online():
    hidden, weight = _make_inputs(rows=7, hidden_size=96, vocab_size=677)
    embed_weight = _make_embed_weight(vocab_size=677, embed_size=48,
                                      seed=20260642)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 1.3, 0.9, 0.8], device="cuda",
        dtype=torch.float32,
    )

    expected = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260643,
        soft_embed_chunk_size=113,
    )
    actual = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260643,
        chunk_size=113,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)


def test_diffusion_gemma_cublas_two_pass_chunk_size_invariant():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32,
                                      seed=20260644)
    expected = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=0.7,
        seed=20260645,
        chunk_size=257,
    )

    for chunk_size, gumbel_chunk_size in (
        (1, None),
        (63, -1),
        (128, 0),
        (128, 64),
    ):
        actual = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=0.7,
            seed=20260645,
            chunk_size=chunk_size,
            gumbel_chunk_size=gumbel_chunk_size,
            gumbel_scratch_mib=1,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-4,
                                   atol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-3,
                                   atol=1e-3)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=2e-2,
                                   atol=2e-2)


def test_diffusion_gemma_cublas_two_pass_temp_zero_is_deterministic_one_hot():
    hidden, weight = _make_inputs(rows=4, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32,
                                      seed=20260646)
    temperature = torch.tensor([0.0, 0.7, 0.0, 1.1], device="cuda",
                               dtype=torch.float32)

    actual = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260647,
        chunk_size=64,
    )
    zero_rows = temperature == 0
    torch.testing.assert_close(actual[2][zero_rows], actual[3][zero_rows],
                               rtol=0, atol=0)
    torch.testing.assert_close(actual[1][zero_rows],
                               torch.zeros_like(actual[1][zero_rows]),
                               rtol=0, atol=0)
    torch.testing.assert_close(
        actual[4][zero_rows],
        embed_weight[actual[3][zero_rows]].float(),
        rtol=0,
        atol=0,
    )


def test_diffusion_gemma_cublas_two_pass_matches_materialized_fp32_reference():
    hidden, weight = _make_inputs(rows=6, hidden_size=96, vocab_size=389)
    embed_weight = _make_embed_weight(vocab_size=389, embed_size=40,
                                      seed=20260648)
    temperature = 0.7

    actual = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260649,
        chunk_size=97,
    )
    expected_lse, expected_entropy, expected_argmax = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=temperature
    )
    expected_sample = _reference_gumbel_sample(
        hidden, weight, softcap=30.0, temperature=temperature, seed=20260649
    )
    expected_soft = _reference_soft_embeds_fp32(
        hidden, weight, embed_weight, softcap=30.0, temperature=temperature
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=1e-3,
                               atol=1e-3)
    torch.testing.assert_close(actual[2], expected_sample, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected_argmax, rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected_soft, rtol=5e-2, atol=5e-2)


def test_diffusion_gemma_cublas_two_pass_row_seed_offsets_are_reorder_stable():
    hidden, weight = _make_inputs(rows=8, hidden_size=80, vocab_size=541)
    embed_weight = _make_embed_weight(vocab_size=541, embed_size=32,
                                      seed=20260650)
    row_seed_offsets = torch.tensor([11, 3, 20, 7, 15, 99, 2, 42],
                                    device="cuda", dtype=torch.int64)
    temperature = torch.full((8,), 0.7, device="cuda", dtype=torch.float32)
    expected = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=temperature,
        seed=20260651, chunk_size=113, row_seed_offsets=row_seed_offsets,
    )

    perm = torch.tensor([5, 0, 7, 1, 6, 2, 4, 3], device="cuda")
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device="cuda")
    actual = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden[perm], weight, embed_weight, softcap=30.0,
        temperature=temperature[perm], seed=20260651, chunk_size=113,
        row_seed_offsets=row_seed_offsets[perm],
    )

    for idx in (0, 1, 2, 3, 4):
        tol = {0: (3e-4, 2e-3), 1: (1e-3, 1e-3), 4: (2e-2, 2e-2)}.get(idx, (0, 0))
        torch.testing.assert_close(actual[idx][inv], expected[idx],
                                   rtol=tol[0], atol=tol[1])


def test_diffusion_gemma_row_chunked_sample_soft_embeds_matches_cublas():
    hidden, weight = _make_inputs(rows=9, hidden_size=96, vocab_size=677)
    embed_weight = _make_embed_weight(vocab_size=677, embed_size=48,
                                      seed=20260652)
    temperature = torch.tensor(
        [0.3, 0.5, 0.7, 1.0, 1.3, 0.9, 0.8, 0.0, 1.1],
        device="cuda",
        dtype=torch.float32,
    )
    row_seed_offsets = torch.tensor([8, 1, 7, 3, 5, 11, 13, 17, 19],
                                    device="cuda", dtype=torch.int64)

    expected = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260653,
        chunk_size=113,
        row_seed_offsets=row_seed_offsets,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260653,
        row_chunk_size=4,
        row_seed_offsets=row_seed_offsets,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)


def test_diffusion_gemma_row_chunked_chunk_size_invariant():
    hidden, weight = _make_inputs(rows=7, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32,
                                      seed=20260654)
    temperature = torch.tensor([0.3, 0.5, 0.7, 1.0, 0.0, 0.9, 1.1],
                               device="cuda", dtype=torch.float32)
    expected = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260655,
        row_chunk_size=7,
    )

    for row_chunk_size in (1, 2, 3):
        actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
            hidden,
            weight,
            embed_weight,
            softcap=30.0,
            temperature=temperature,
            seed=20260655,
            row_chunk_size=row_chunk_size,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-4,
                                   atol=2e-3)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-3,
                                   atol=1e-3)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        torch.testing.assert_close(actual[4], expected[4], rtol=2e-2,
                                   atol=2e-2)


def test_diffusion_gemma_row_chunked_temp_zero_matches_online_boundary():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=32,
                                      seed=20260656)
    temperature = torch.tensor([0.0, 0.7, 0.0, 1.1, 0.0], device="cuda",
                               dtype=torch.float32)

    expected = diffusion_gemma_softcap_online_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260657,
        soft_embed_chunk_size=63,
    )
    actual = diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260657,
        row_chunk_size=2,
    )

    zero_rows = temperature == 0
    torch.testing.assert_close(actual[2][zero_rows], actual[3][zero_rows],
                               rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4][zero_rows],
                               expected[4][zero_rows],
                               rtol=2e-2, atol=2e-2)


def test_diffusion_gemma_triton_single_pass_matches_reference():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=48,
                                      seed=20260640)
    temperature = torch.full((5,), 0.7, device="cuda", dtype=torch.float32)

    actual = diffusion_gemma_softcap_triton_single_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260641,
        block_m=2,
        block_n=64,
        block_k=32,
        block_e=32,
    )
    expected_lse, expected_entropy, expected_argmax = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=0.7
    )
    expected_soft = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual[3], expected_argmax, rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected_soft, rtol=2e-2, atol=2e-2)
    assert ((actual[2] >= 0) & (actual[2] < weight.shape[0])).all()


def test_diffusion_gemma_soft_embed_from_lse_matches_materialized_reference():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=48,
                                      seed=20260662)
    temperature = torch.full((5,), 0.7, device="cuda", dtype=torch.float32)
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / 30.0) * 30.0 / temperature[:, None]
    lse = scaled.logsumexp(dim=-1)
    expected = (
        scaled.softmax(dim=-1).to(embed_weight.dtype) @ embed_weight
    ).float()

    actual = diffusion_gemma_softcap_soft_embed_from_lse(
        hidden,
        weight,
        embed_weight,
        lse,
        softcap=30.0,
        temperature=temperature,
        block_m=4,
        block_n=64,
        block_k=32,
        block_e=32,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert actual.dtype == torch.float32


def test_row_chunked_soft_embed_from_lse_matches_reference():
    hidden, weight = _make_inputs(rows=9, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=48,
                                      seed=20260663)
    temperature = torch.linspace(0.6, 1.1, 9, device="cuda",
                                 dtype=torch.float32)
    logits = hidden @ weight.t()
    scaled = torch.tanh(logits.float() / 30.0) * 30.0 / temperature[:, None]
    lse = scaled.logsumexp(dim=-1)
    expected = (
        scaled.softmax(dim=-1).to(embed_weight.dtype) @ embed_weight
    ).float()

    actual = diffusion_gemma_softcap_row_chunked_soft_embed_from_lse(
        hidden,
        weight,
        embed_weight,
        lse,
        softcap=30.0,
        temperature=temperature,
        row_chunk_size=3,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert actual.dtype == torch.float32


def test_row_chunked_soft_embed_from_lse_scalar_temp_large_chunk():
    hidden, weight = _make_inputs(rows=4, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=48,
                                      seed=20260664)
    temperature = torch.full((4,), 0.7, device="cuda", dtype=torch.float32)
    scores = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    scores = torch.tanh(scores / 30.0) * 30.0 / temperature[:, None]
    lse = scores.logsumexp(dim=-1)
    expected = torch.mm(scores.softmax(dim=-1).to(embed_weight.dtype),
                        embed_weight, out_dtype=torch.float32)

    actual = diffusion_gemma_softcap_row_chunked_soft_embed_from_lse(
        hidden,
        weight,
        embed_weight,
        lse,
        softcap=30.0,
        temperature=0.7,
        row_chunk_size=32,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_diffusion_gemma_row_chunked_soft_embed_from_lse_handles_empty_rows():
    hidden = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((257, 64), device="cuda", dtype=torch.bfloat16)
    embed_weight = torch.empty((257, 48), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((0,), device="cuda", dtype=torch.float32)

    actual = diffusion_gemma_softcap_row_chunked_soft_embed_from_lse(
        hidden, weight, embed_weight, lse, softcap=30.0, temperature=0.7
    )

    assert actual.shape == (0, 48)
    assert actual.dtype == torch.float32


def test_diffusion_gemma_soft_embed_from_lse_handles_empty_rows():
    hidden = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((257, 64), device="cuda", dtype=torch.bfloat16)
    embed_weight = torch.empty((257, 48), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((0,), device="cuda", dtype=torch.float32)

    actual = diffusion_gemma_softcap_soft_embed_from_lse(
        hidden, weight, embed_weight, lse, softcap=30.0, temperature=0.7
    )

    assert actual.shape == (0, 48)
    assert actual.dtype == torch.float32


def test_diffusion_gemma_triton_sample_soft_embeds_matches_reference():
    hidden, weight = _make_inputs(rows=5, hidden_size=64, vocab_size=257)
    embed_weight = _make_embed_weight(vocab_size=257, embed_size=48,
                                      seed=20260636)
    temperature = torch.full((5,), 0.7, device="cuda", dtype=torch.float32)

    actual = diffusion_gemma_softcap_triton_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260637,
        block_m=4,
        block_n=64,
        block_k=32,
        block_e=32,
    )
    expected_lse, expected_entropy, expected_argmax = _reference_softcap_reductions(
        hidden, weight, softcap=30.0, temperature=0.7
    )
    expected_soft = _reference_soft_embeds(
        hidden, weight, embed_weight, softcap=30.0, temperature=0.7
    )

    torch.testing.assert_close(actual[0], expected_lse, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=1e-3, atol=1e-3)
    _assert_selected_token_is_near_materialized_max(
        hidden, weight, actual[3], softcap=30.0, temperature=0.7
    )
    torch.testing.assert_close(actual[3], expected_argmax, rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected_soft, rtol=2e-2, atol=2e-2)
    assert actual[2].shape == expected_argmax.shape
    assert ((actual[2] >= 0) & (actual[2] < weight.shape[0])).all()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Triton projection semantics can drift beyond the sampler entropy "
        "threshold at large vocab/low temperature; keep triton_full "
        "experimental, not exact."
    ),
)
def test_diffusion_gemma_triton_sample_large_vocab_low_temp_matches_cublas_greedy():
    # Council gate: exercise the full-output Triton path at larger vocab and
    # low temperature so softcap/tail-mass behavior is tested beyond the tiny
    # unit shapes. Sample tokens are not compared because Triton uses a
    # prototype inline RNG stream; greedy/LSE/entropy/soft_embed must match.
    hidden, weight = _make_inputs(rows=3, hidden_size=64, vocab_size=32768)
    embed_weight = _make_embed_weight(vocab_size=32768, embed_size=32,
                                      seed=20260658)
    temperature = torch.full((3,), 0.2, device="cuda", dtype=torch.float32)

    expected = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260659,
        chunk_size=4096,
    )
    actual = diffusion_gemma_softcap_triton_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260659,
        block_m=4,
        block_n=128,
        block_k=32,
        block_e=32,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)
    assert ((actual[2] >= 0) & (actual[2] < weight.shape[0])).all()


def test_diffusion_gemma_triton_sample_greedy_stable_when_rng_seed_aliases():
    # The Triton prototype truncates seed internally for its sampling RNG.
    # Document that sampled tokens are backend-specific while deterministic
    # greedy state remains aligned with the cublas/row_chunked reference.
    hidden, weight = _make_inputs(rows=4, hidden_size=64, vocab_size=4097)
    embed_weight = _make_embed_weight(vocab_size=4097, embed_size=32,
                                      seed=20260660)
    temperature = torch.full((4,), 0.7, device="cuda", dtype=torch.float32)
    seed = 20260661 + 65536

    expected = diffusion_gemma_softcap_cublas_two_pass_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=seed,
        chunk_size=1024,
    )
    actual = diffusion_gemma_softcap_triton_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=seed,
        block_m=4,
        block_n=128,
        block_k=32,
        block_e=32,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=3e-4, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-2, atol=2e-2)
    assert ((actual[2] >= 0) & (actual[2] < weight.shape[0])).all()


def test_diffusion_gemma_triton_sample_temperature_zero_is_greedy():
    hidden, weight = _make_inputs(rows=4, hidden_size=48, vocab_size=193)
    embed_weight = _make_embed_weight(vocab_size=193, embed_size=32,
                                      seed=20260638)
    temperature = torch.zeros((4,), device="cuda", dtype=torch.float32)

    actual = diffusion_gemma_softcap_triton_sample_soft_embeds(
        hidden,
        weight,
        embed_weight,
        softcap=30.0,
        temperature=temperature,
        seed=20260639,
        block_m=4,
        block_n=64,
        block_k=32,
        block_e=32,
    )
    torch.testing.assert_close(actual[2], actual[3], rtol=0, atol=0)


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
