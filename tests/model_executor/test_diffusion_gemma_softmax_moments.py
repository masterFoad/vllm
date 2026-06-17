# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.models import diffusion_gemma


def _materialized_softmax_moments(
    scaled: torch.Tensor,
    embed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    soft_embeds = (probs.to(embed_weight.dtype) @ embed_weight).float()
    return entropy, soft_embeds


def _materialized_sample_step_reference(
    logits: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_idx: torch.Tensor,
    all_slots: torch.Tensor,
    valid_canvas_len: torch.Tensor,
    canvas: torch.Tensor,
    argmax_canvas: torch.Tensor,
    step_tensor: torch.Tensor,
    is_encoder_phase: torch.Tensor,
    confident_tensor: torch.Tensor,
    sc_embeds: torch.Tensor,
    embed_weight: torch.Tensor,
    normalizer: torch.Tensor,
    history: torch.Tensor,
    history_len_tensor: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    draft_tokens: torch.Tensor,
    max_denoising_steps: float,
    t_min: float,
    t_max: float,
    confidence_threshold: float,
    vocab_size: int,
    CL: int,
    ST: int,
    entropy_bound: float,
    return_scaled: bool,
) -> torch.Tensor:
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    sampled.zero_()
    num_sampled.zero_()

    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    logits_3d = logits.reshape(num_decode, CL, -1)
    scaled = logits_3d.to(dtype=torch.float32, copy=True)
    scaled.div_(temp[:, None, None].clamp(min=1e-10))

    noisy = torch.empty_like(scaled)
    noisy.exponential_()
    noisy.log_()
    noisy.neg_()
    noisy.mul_((temp[:, None, None] > 0).float())
    noisy.add_(scaled)
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    argmax_tokens = (
        scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    )

    token_entropy, soft_embeds = _materialized_softmax_moments(
        scaled, embed_weight
    )
    mean_entropy = token_entropy.mean(dim=-1)
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    is_commit = is_encoder_phase[decode_slots]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()

    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    random_tokens = torch.randint(
        0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype
    )

    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    sampled[decode_idx] = argmax_canvas[decode_slots].to(
        sampled.dtype
    ) * is_commit.unsqueeze(1).to(sampled.dtype)
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
        num_sampled.dtype
    )

    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps
    )
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    sc_embeds[decode_slots] = (soft_embeds * normalizer * sc_keep).to(
        sc_embeds.dtype
    )

    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )

    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return scaled if return_scaled else scaled.new_empty((0,))


def _sample_step_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    num_decode = 3
    max_num_reqs = 4
    CL = 5
    ST = 3
    vocab_size = 37
    hidden = 7

    logits = torch.randn(num_decode * CL, vocab_size, dtype=torch.bfloat16) * 2
    # Exercise zero-padded tail rows for valid_canvas_len < CL.
    logits = logits.float()
    logits[CL + 3 : CL + 5].zero_()
    logits = logits.bfloat16()

    return {
        "logits": logits,
        "decode_slots": torch.tensor([0, 1, 2], dtype=torch.long),
        "decode_idx": torch.tensor([0, 1, 2], dtype=torch.long),
        "all_slots": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "valid_canvas_len": torch.tensor([5, 3, 4], dtype=torch.long),
        "canvas": torch.randint(0, vocab_size, (max_num_reqs, CL), dtype=torch.long),
        "argmax_canvas": torch.randint(
            0, vocab_size, (max_num_reqs, CL), dtype=torch.long
        ),
        "step_tensor": torch.tensor([1, 3, 2, 0], dtype=torch.long),
        "is_encoder_phase": torch.tensor([False, False, True, False]),
        "confident_tensor": torch.zeros(max_num_reqs, dtype=torch.bool),
        "sc_embeds": torch.zeros(max_num_reqs, CL, hidden, dtype=torch.bfloat16),
        "embed_weight": torch.randn(vocab_size, hidden, dtype=torch.bfloat16),
        "normalizer": torch.tensor(0.75, dtype=torch.float32),
        "history": torch.randint(
            0, vocab_size, (max_num_reqs, ST, CL), dtype=torch.long
        ),
        "history_len_tensor": torch.tensor([1, 2, 0, 0], dtype=torch.long),
        "sampled": torch.empty(max_num_reqs, CL, dtype=torch.long),
        "num_sampled": torch.empty(max_num_reqs, dtype=torch.long),
        "draft_tokens": torch.empty(max_num_reqs, CL + 2, dtype=torch.long),
    }


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in state.items()}


class _FakeUvaBackedTensor:
    def __init__(self, size, dtype: torch.dtype, *_, **__):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        self.gpu = self.cpu[:n] if n is not None else self.cpu
        return self.gpu


class _FakeSamplingStates:
    def __init__(self, max_num_logprobs: int):
        self._max_num_logprobs = max_num_logprobs

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return self._max_num_logprobs


def _confidence_threshold_for_state(state: dict[str, torch.Tensor]) -> float:
    num_decode = state["decode_slots"].shape[0]
    CL = 5
    max_denoising_steps = 4.0
    t_min = 0.25
    t_max = 1.0
    steps_f = state["step_tensor"][state["decode_slots"]].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)
    scaled = state["logits"].reshape(num_decode, CL, -1).float()
    scaled = scaled / temp[:, None, None].clamp(min=1e-10)
    entropy, _ = _materialized_softmax_moments(scaled, state["embed_weight"])
    return float(entropy.mean(dim=-1)[0].item() + 1e-4)


def _run_sample_step(
    fn,
    state: dict[str, torch.Tensor],
    *,
    confidence_threshold: float,
    t_min: float = 0.25,
    t_max: float = 1.0,
    entropy_bound: float = 0.25,
    return_scaled: bool = True,
) -> torch.Tensor:
    return fn(
        state["logits"],
        state["decode_slots"],
        state["decode_idx"],
        state["all_slots"],
        state["valid_canvas_len"],
        state["canvas"],
        state["argmax_canvas"],
        state["step_tensor"],
        state["is_encoder_phase"],
        state["confident_tensor"],
        state["sc_embeds"],
        state["embed_weight"],
        state["normalizer"],
        state["history"],
        state["history_len_tensor"],
        state["sampled"],
        state["num_sampled"],
        state["draft_tokens"],
        max_denoising_steps=4.0,
        t_min=t_min,
        t_max=t_max,
        confidence_threshold=confidence_threshold,
        vocab_size=state["embed_weight"].shape[0],
        CL=state["canvas"].shape[1],
        ST=state["history"].shape[1],
        entropy_bound=entropy_bound,
        return_scaled=return_scaled,
    )


def test_softmax_moments_match_materialized_reference_with_bf16_embed(
    monkeypatch,
):
    torch.manual_seed(0)
    scaled = torch.randn(3, 5, 37, dtype=torch.float32) * 3
    embed_weight = torch.randn(37, 11, dtype=torch.bfloat16)

    entropy_ref, soft_embeds_ref = _materialized_softmax_moments(
        scaled, embed_weight
    )

    for chunk_size in (1, 7, 1024):
        monkeypatch.setattr(
            diffusion_gemma, "_DIFFUSION_GEMMA_SC_CHUNK_SIZE", chunk_size
        )
        entropy, soft_embeds = diffusion_gemma._compute_softmax_moments(
            scaled, embed_weight, vocab_size=scaled.shape[-1]
        )

        torch.testing.assert_close(entropy, entropy_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            soft_embeds, soft_embeds_ref, rtol=2e-2, atol=2e-2
        )
        assert soft_embeds.dtype == torch.float32


def test_softmax_moments_accumulate_soft_embeddings_in_fp32(monkeypatch):
    torch.manual_seed(7)
    scaled = torch.randn(2, 3, 257, dtype=torch.float32) * 4
    embed_weight = torch.randn(257, 19, dtype=torch.bfloat16)
    chunk_size = 31
    monkeypatch.setattr(
        diffusion_gemma, "_DIFFUSION_GEMMA_SC_CHUNK_SIZE", chunk_size
    )

    _, soft_embeds_ref = _materialized_softmax_moments(scaled, embed_weight)
    _, soft_embeds = diffusion_gemma._compute_softmax_moments(
        scaled, embed_weight, vocab_size=scaled.shape[-1]
    )

    lse = scaled.logsumexp(dim=-1)
    old_bf16_accum = torch.zeros(
        (*scaled.shape[:-1], embed_weight.shape[1]), dtype=torch.bfloat16
    )
    for start in range(0, scaled.shape[-1], chunk_size):
        scaled_chunk = scaled[..., start : start + chunk_size]
        probs_chunk = (scaled_chunk - lse[..., None]).exp()
        old_bf16_accum.add_(
            probs_chunk.to(embed_weight.dtype)
            @ embed_weight[start : start + chunk_size]
        )

    new_err = (soft_embeds - soft_embeds_ref).abs().max().item()
    old_err = (old_bf16_accum.float() - soft_embeds_ref).abs().max().item()
    assert soft_embeds.dtype == torch.float32
    assert new_err < 2e-2
    assert old_err > 1e-3
    assert new_err < old_err



def test_sample_step_matches_materialized_reference(monkeypatch):
    monkeypatch.setattr(diffusion_gemma, "_DIFFUSION_GEMMA_SC_CHUNK_SIZE", 11)
    base_state = _sample_step_state()
    confidence_threshold = _confidence_threshold_for_state(base_state)
    opt_state = _clone_state(base_state)
    ref_state = _clone_state(base_state)

    eager_step = diffusion_gemma._compiled_sample_step.__wrapped__

    torch.manual_seed(777)
    opt_scaled = _run_sample_step(
        eager_step,
        opt_state,
        confidence_threshold=confidence_threshold,
    )
    torch.manual_seed(777)
    ref_scaled = _run_sample_step(
        _materialized_sample_step_reference,
        ref_state,
        confidence_threshold=confidence_threshold,
    )

    torch.testing.assert_close(opt_scaled, ref_scaled, rtol=0, atol=0)
    for name in (
        "canvas",
        "argmax_canvas",
        "step_tensor",
        "is_encoder_phase",
        "confident_tensor",
        "history",
        "history_len_tensor",
        "sampled",
        "num_sampled",
        "draft_tokens",
    ):
        torch.testing.assert_close(opt_state[name], ref_state[name], rtol=0, atol=0)
    torch.testing.assert_close(
        opt_state["sc_embeds"], ref_state["sc_embeds"], rtol=1e-2, atol=1e-2
    )

    # Slot 2 entered as a commit row and has valid_canvas_len=4, so only four
    # tokens are reported even though the internal canvas has length five.
    assert opt_state["num_sampled"][2].item() == 4
    # Slot 0 remains a denoise row, so self-conditioning is populated there.
    assert opt_state["sc_embeds"][0].abs().sum().item() > 0
    # Commit and converged rows do not carry self-conditioning into the next step.
    assert opt_state["sc_embeds"][1].abs().sum().item() == 0
    assert opt_state["sc_embeds"][2].abs().sum().item() == 0


def test_compiled_sample_step_matches_eager_greedy_path(monkeypatch):
    monkeypatch.setattr(diffusion_gemma, "_DIFFUSION_GEMMA_SC_CHUNK_SIZE", 11)
    base_state = _sample_step_state()
    base_state["is_encoder_phase"].zero_()
    base_state["step_tensor"].zero_()
    eager_state = _clone_state(base_state)
    compiled_state = _clone_state(base_state)

    eager_step = diffusion_gemma._compiled_sample_step.__wrapped__

    eager_scaled = _run_sample_step(
        eager_step,
        eager_state,
        confidence_threshold=-math.inf,
        t_min=0.0,
        t_max=0.0,
        entropy_bound=math.inf,
    )
    compiled_scaled = _run_sample_step(
        diffusion_gemma._compiled_sample_step,
        compiled_state,
        confidence_threshold=-math.inf,
        t_min=0.0,
        t_max=0.0,
        entropy_bound=math.inf,
    )

    torch.testing.assert_close(compiled_scaled, eager_scaled, rtol=0, atol=0)
    for name in (
        "canvas",
        "argmax_canvas",
        "step_tensor",
        "is_encoder_phase",
        "confident_tensor",
        "history",
        "history_len_tensor",
        "sampled",
        "num_sampled",
        "draft_tokens",
        "sc_embeds",
    ):
        torch.testing.assert_close(
            compiled_state[name], eager_state[name], rtol=0, atol=0
        )


def test_diffusion_sampler_requests_scaled_logits_only_for_logprobs(monkeypatch):
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

    seen_return_scaled: list[bool] = []

    def fake_compiled_sample_step(
        logits,
        decode_slots,
        decode_idx,
        all_slots,
        valid_canvas_len,
        canvas,
        argmax_canvas,
        step_tensor,
        is_encoder_phase,
        confident_tensor,
        sc_embeds,
        embed_weight,
        normalizer,
        history,
        history_len_tensor,
        sampled,
        num_sampled,
        draft_tokens,
        max_denoising_steps,
        t_min,
        t_max,
        confidence_threshold,
        vocab_size,
        CL,
        ST,
        entropy_bound,
        return_scaled,
    ):
        seen_return_scaled.append(return_scaled)
        sampled.zero_()
        num_sampled.zero_()
        draft_tokens[all_slots, :CL] = canvas[all_slots]
        if return_scaled:
            return logits.reshape(decode_slots.shape[0], CL, -1).float()
        return logits.new_empty((0,), dtype=torch.float32)

    monkeypatch.setattr(
        diffusion_gemma, "_compiled_sample_step", fake_compiled_sample_step
    )

    def run_case(max_num_logprobs: int):
        states = diffusion_gemma.DiffusionGemmaRequestStates(
            max_num_reqs=2,
            canvas_length=3,
            vocab_size=7,
            max_denoising_steps=4,
            device=torch.device("cpu"),
            hidden_size=5,
            stability_threshold=2,
        )
        sampler = SimpleNamespace(
            sampling_states=_FakeSamplingStates(max_num_logprobs),
            req_states=SimpleNamespace(
                draft_tokens=torch.zeros(2, 3, dtype=torch.long)
            ),
        )
        diffusion_sampler = diffusion_gemma.DiffusionSampler(
            sampler=sampler,
            diffusion_config=SimpleNamespace(canvas_length=3),
            vocab_size=7,
            diffusion_states=states,
            confidence_threshold=1.0,
            t_min=0.0,
            t_max=1.0,
            entropy_bound=1.0,
            embed_weight=torch.randn(7, 5),
            normalizer=torch.tensor(1.0),
        )
        input_batch = SimpleNamespace(
            num_reqs=2,
            num_draft_tokens=1,
            idx_mapping_np=np.array([0, 1], dtype=np.int64),
            idx_mapping=torch.tensor([0, 1], dtype=torch.long),
            cu_num_logits_np=np.array([0, 3, 6], dtype=np.int64),
            query_start_loc_np=np.array([0, 3, 6], dtype=np.int64),
            query_start_loc=torch.tensor([0, 3, 6], dtype=torch.long),
        )
        output = diffusion_sampler(torch.randn(6, 7), input_batch)
        assert output.logprobs_tensors is None

    run_case(max_num_logprobs=-1)
    run_case(max_num_logprobs=0)
    assert seen_return_scaled == [False, True]



def test_sample_step_reuses_fp32_logits_for_scaled_output():
    state = _sample_step_state()
    # DiffusionGemma final-logit softcap hands the sampler an fp32 logits tensor.
    # The sampler owns that tensor after model_runner passes it in, so eager
    # scaling should reuse the buffer instead of allocating another full-vocab
    # fp32 copy.
    state["logits"] = state["logits"].float()
    expected = state["logits"].reshape(
        state["decode_slots"].shape[0], state["canvas"].shape[1], -1
    ).clone()
    confidence_threshold = _confidence_threshold_for_state(state)
    eager_step = diffusion_gemma._compiled_sample_step.__wrapped__

    scaled = _run_sample_step(
        eager_step,
        state,
        confidence_threshold=confidence_threshold,
        t_min=1.0,
        t_max=1.0,
    )

    assert scaled.data_ptr() == state["logits"].data_ptr()
    torch.testing.assert_close(scaled, expected, rtol=0, atol=0)


def test_sample_step_temperature_zero_is_greedy(monkeypatch):
    monkeypatch.setattr(diffusion_gemma, "_DIFFUSION_GEMMA_SC_CHUNK_SIZE", 13)
    state = _sample_step_state()
    state["decode_slots"] = torch.tensor([0], dtype=torch.long)
    state["decode_idx"] = torch.tensor([0], dtype=torch.long)
    state["valid_canvas_len"] = torch.tensor([5], dtype=torch.long)
    state["is_encoder_phase"].zero_()
    state["step_tensor"].zero_()
    logits = state["logits"][: state["canvas"].shape[1]]
    state["logits"] = logits

    eager_step = diffusion_gemma._compiled_sample_step.__wrapped__
    _run_sample_step(
        eager_step,
        state,
        confidence_threshold=math.inf,
        t_min=0.0,
        t_max=0.0,
        entropy_bound=math.inf,
        return_scaled=False,
    )

    expected = logits.float().reshape(1, state["canvas"].shape[1], -1).argmax(dim=-1)
    torch.testing.assert_close(state["canvas"][0:1], expected, rtol=0, atol=0)
    torch.testing.assert_close(state["argmax_canvas"][0:1], expected, rtol=0, atol=0)


def test_exponential_noise_gumbel_max_matches_softmax_distribution():
    torch.manual_seed(0)
    num_samples = 200_000
    logits = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.25, 0.75, 1.0, 2.0])
    vocab_size = logits.numel()
    target = logits.softmax(dim=-1)

    exp_noise = torch.empty(num_samples, vocab_size)
    exp_noise.exponential_()
    new_samples = (logits - exp_noise.log()).argmax(dim=-1)
    new_freq = torch.bincount(new_samples, minlength=vocab_size).float() / num_samples
    new_tv = 0.5 * (new_freq - target).abs().sum().item()

    torch.manual_seed(0)
    uniforms = torch.rand(num_samples, vocab_size).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    old_gumbels = -torch.log(-torch.log(uniforms))
    old_samples = (logits + old_gumbels).argmax(dim=-1)
    old_freq = torch.bincount(old_samples, minlength=vocab_size).float() / num_samples
    old_tv = 0.5 * (old_freq - target).abs().sum().item()

    tv_bound = 6.0 * math.sqrt(vocab_size / num_samples)
    assert new_tv < tv_bound
    assert old_tv < tv_bound
    assert new_tv <= old_tv + tv_bound / 4.0
