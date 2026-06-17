# DiffusionGemma adaptive sampler focused diff

This is the review-oriented diff report for branch
`masterfoad/diffusion-gemma-adaptive-sampler-clean-20260617T143626Z`.
It is meant to be read against current vLLM `main`, not against the stale fork
history that previously made GitHub compare show thousands of unrelated files.

## Diff identity

- Base: upstream/fork `main` at `46f74e144`
- Feature commit: `035ff1108` (`Prevent DiffusionGemma sampler spikes from crashing serving`)
- Focused range:

```bash
git diff 46f74e144..035ff1108
git diff --stat 46f74e144..035ff1108
```

At the time this report was written, the branch is exactly **1 commit ahead of
current vLLM main**.

## One-sentence purpose

Prevent DiffusionGemma serving from crashing under high decode-row pressure by
making the model reserve memory for unprofiled sampler spikes and by adding
bounded streamed sampler fallbacks, while keeping the fast materialized sampler
for ordinary low-pressure cases.

## What changed, by file

| File | Type | Review focus |
| --- | --- | --- |
| `vllm/model_executor/models/diffusion_gemma.py` | Core behavior | Adds DiffusionGemma-specific sampler mode selection, row-pressure thresholds, memory reserve estimate, decode-batch diagnostics, and calls into streamed fallback paths. |
| `vllm/model_executor/models/diffusion_gemma_fused_lse.py` | New helper module | Implements exact chunked/streamed LSE, entropy/moment, Gumbel, argmax, row-chunk, and experimental Triton helpers used by tests/benchmarks and gated serving paths. |
| `vllm/envs.py` | Config | Adds DiffusionGemma-only env knobs for reserve sizing, streamed backend selection, row/vocab chunk sizes, experimental Triton gate, and decode-batch logging. |
| `vllm/v1/worker/gpu_worker.py` | KV sizing hook | Subtracts model-reported non-KV runtime reserve from available KV memory so vLLM does not overfill the GPU and then die on unprofiled decode sampler spikes. |
| `vllm/v1/worker/gpu/model_runner.py` | Hook plumbing | Forwards model-reported extra non-KV memory from the loaded model to the worker. |
| `vllm/v1/worker/gpu/model_states/interface.py` | Interface | Adds default `get_extra_non_kv_cache_memory_bytes() -> int` hook returning `0`, so non-DiffusionGemma models are unchanged. |
| `tests/model_executor/test_diffusion_gemma_fused_lse.py` | Tests | Covers streamed LSE/moment/argmax/Gumbel/soft-embed numerics, edge cases, and experimental helper behavior against materialized references. |
| `tests/model_executor/test_diffusion_gemma_softmax_moments.py` | Tests | Locks the moment math and bf16/fp32 accumulation behavior that fixed earlier correctness drift. |
| `tests/v1/worker/test_diffusion_sampler_memory_reserve.py` | Tests | Verifies the model-reported non-KV reserve hook is subtracted from KV-cache sizing and rejects over-reservation. |
| `benchmarks/kernels/benchmark_diffusion_gemma_soft_embed_pass2.py` | Benchmark helper | Isolated helper for pass-2/soft-embed recompute experiments; not a serving default. |
| `docs/design/diffusion_gemma_adaptive_sampler_report.md` | Report | Full experiment log, benchmark tables, failed paths, handoff, skills used, and next-step plan. |

## Runtime behavior summary

Default upstream behavior is preserved unless DiffusionGemma-specific knobs are
enabled. The intended safe production-style configuration from the validation
runs is the reserve-auto/adaptive setup:

```bash
export VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_MIB=auto
export VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_SCALE=1.1
export VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1
export VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto
```

Behavior under that setup:

1. Low row pressure stays on the fast materialized sampler path.
2. High row pressure can switch to bounded streamed fallback paths.
3. vLLM reserves model-reported non-KV memory before sizing KV cache, preventing
   the known unprofiled sampler spike from consuming all remaining headroom.

## Important non-goals / caveats

This branch should **not** be described as any of the following yet:

- a general vLLM throughput speedup;
- a lower-latency improvement when the materialized sampler already fits;
- proof that startup KV-cache tokens increase at fixed utilization;
- a PR-ready Triton fused kernel;
- validated tensor-parallel (`TP>1`) behavior;
- protection for every logprobs/high-pressure path.

The strongest validated claim is narrower and cleaner:

> On A100 ShareGPT pressure tests, the adaptive reserve path prevented
> DiffusionGemma OOM/server-death cases seen in the materialized baseline while
> keeping the fast materialized path for lower row pressure.

## Why the fork compare was noisy before

The fork `main` branch was stale, so GitHub compared the feature branch against
an old fork base and reported thousands of unrelated upstream changes. The fix
is to fast-forward `masterFoad/vllm:main` to current `vllm-project/vllm:main` and
keep the feature branch rebased on that same commit.

Expected focused shape after sync:

```text
main:    46f74e144  [latest upstream main at sync time]
feature: 035ff1108  one DiffusionGemma commit on top
```

So reviewers should compare:

```bash
git diff masterFoad/main...masterFoad/masterfoad/diffusion-gemma-adaptive-sampler-clean-20260617T143626Z
```

or locally:

```bash
git fetch origin main
git fetch fork main masterfoad/diffusion-gemma-adaptive-sampler-clean-20260617T143626Z
git diff fork/main..fork/masterfoad/diffusion-gemma-adaptive-sampler-clean-20260617T143626Z
```

## Verification already run for this clean branch

Local/static checks on the clean worktree:

```bash
python3 -m py_compile \
  benchmarks/kernels/benchmark_diffusion_gemma_soft_embed_pass2.py \
  vllm/model_executor/models/diffusion_gemma.py \
  vllm/model_executor/models/diffusion_gemma_fused_lse.py \
  vllm/envs.py \
  vllm/v1/worker/gpu_worker.py \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/v1/worker/gpu/model_states/interface.py \
  tests/model_executor/test_diffusion_gemma_fused_lse.py \
  tests/model_executor/test_diffusion_gemma_softmax_moments.py \
  tests/v1/worker/test_diffusion_sampler_memory_reserve.py

ruff check <same file list>
git diff --check origin/main..HEAD
```

Result: all passed.

Local pytest was not used as final evidence because the local laptop environment
has a Transformers/vLLM version mismatch. OpenShift/A100 remains the source of
truth for GPU correctness and serving performance; see the full report for the
artifact map and benchmark tables.

## Review checklist

- Confirm the reserve hook is model-scoped and does not affect other models when
  they return the default `0` extra memory.
- Confirm the DiffusionGemma env defaults do not change serving behavior unless
  explicitly enabled.
- Confirm `triton_full` remains gated as experimental.
- Confirm row-chunk fallback is not the default low-pressure path.
- Confirm benchmark claims are framed as OOM/capacity stabilization, not a broad
  speedup.
- Before upstream PR: rerun GPU tests in vLLM CI-compatible environment, validate
  logprobs path separately, and test/guard TP>1.
