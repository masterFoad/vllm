# DiffusionGemma adaptive sampler memory report

Date: 2026-06-16
Model: `google/diffusiongemma-26B-A4B-it`
Target environment: OpenShift `wdu-research`, A100-class GPU, vLLM OpenAI server
Status: local branch only; no PR opened

## Executive summary

We started by trying to reduce DiffusionGemma sampler memory without hurting
serving throughput. The useful result is not a universal speedup. The useful
result is an **adaptive exact sampler backend** that keeps the fast materialized
path for normal batch sizes and switches to bounded row-chunking only when the
number of sampler rows is large enough to risk CUDA OOM.

The headline evidence from OpenShift/A100 testing:

- Normal low-pressure serving: always-on row-chunking is slower than baseline,
  so it should not be enabled unconditionally.
- High-pressure serving with `--max-num-batched-tokens 4096 --max-num-seqs 16`:
  the baseline materialized path OOMed at client concurrency 12 and 16, while
  the adaptive backend completed both runs with zero request errors.
- Correctness tests for the exact helper paths passed on OpenShift:
  `42 passed, 1 xfailed`. The xfail documents the experimental Triton backend;
  it is intentionally not part of the exact serving path.

Honest claim:

> The adaptive backend is an OOM/capacity mitigation for DiffusionGemma's
> sampler. It preserves the fast materialized path below a row threshold and
> falls back to exact row-chunking above that threshold. In the tested
> high-pressure A100 configuration, it avoids OOMs where the baseline fails.

Non-claims:

- It is not a general throughput speedup.
- It does not increase vLLM's startup KV-cache allocation at the tested default
  settings.
- It does not make the experimental Triton fused backend PR-ready.
- It does not address the separate logprobs-on bug class discussed during the
  investigation.

## Why DiffusionGemma sampler memory is special

DiffusionGemma samples over a canvas, not one autoregressive row at a time. With
`canvas_length = 256`, a decode batch with `num_decode` active requests creates
approximately:

```text
sampler_rows = num_decode * canvas_length
```

For `num_decode = 8`, this is `2048` rows. A single full-vocab fp32 tensor is:

```text
sampler_rows * vocab_size * 4 bytes
```

At large vocab sizes this quickly reaches GiB-scale transient allocations. The
sampler needs token entropy, sampled tokens, argmax tokens, and
self-conditioning soft embeddings. A simple materialized implementation can keep
multiple `[sampler_rows, vocab]` tensors live around the same step.

## Terminology used in the benchmark tables

- `c=3`, `c8`, `c12`, `c16`: client concurrency/fanout in the serving benchmark.
  For example, `c12` means the harness keeps up to 12 HTTP requests in flight.
- In the **row-chunk focused c=3 rerun**, each variant ran three timed rounds at
  client concurrency 3. Each round submitted 12 total requests. The table reports
  mean/min/max across those three rounds.
- `tok/s`: aggregate completion tokens per second across successful requests.
- `errors`: request failures observed by the benchmark harness.
- `mem max`: peak GPU memory seen by the harness via coarse server-side polling;
  use it as a pressure signal, not as a precise allocator attribution.
- `sampler_rows`: active decode rows entering the sampler. For the default
  256-token canvas, 3072 rows corresponds to 12 decode requests.

## What changed

Primary files:

- `vllm/model_executor/models/diffusion_gemma.py`
- `vllm/model_executor/models/diffusion_gemma_fused_lse.py`
- `tests/model_executor/test_diffusion_gemma_fused_lse.py`

Key behavior:

- Added exact streamed backend selector:
  `VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND`.
- Added opt-in `auto` backend policy for the streamed sampler:
  - if `sampler_rows <= 2048`, resolve internally to the existing materialized exact path;
  - if `sampler_rows > 2048`, use exact row-chunked sampling;
  - the fallback row chunk defaults to `128` for `auto`.
- Added knobs:
  - `VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER`
  - `VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS`
  - `VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK`
  - `VLLM_DIFFUSION_GEMMA_LOG_DECODE_BATCH`
- Kept `triton_full` behind an explicit experimental gate because it had
  exactness drift in the large-vocab/low-temperature regime.

Important default note: the streamed sampler is fully opt-in. The model keeps the
upstream materialized path unless both of these are set:

```text
VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1
VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto
```

Without `VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1`, the backend selector is not
used. With the streamed sampler enabled but no backend override, the selector
uses `eager`, not `auto`.

## Paths we tried and rejected

### 1. Always-on row-chunking

Always-on row-chunking lowers peak materialization pressure, but it regresses the
normal serving path. That makes it unsuitable as the default.

OpenShift/A100 row-chunk focused c=3 rerun:

| variant | mean tok/s | min tok/s | max tok/s | mean p50 s | mean p95 s | ready MiB | max bench MiB | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 224.70 | 196.33 | 245.35 | 1.07 | 1.57 | 77293 | 77295 | 0 |
| rowchunk_128 | 138.78 | 135.39 | 141.81 | 1.70 | 2.63 | 77293 | 77295 | 0 |
| rowchunk_256 | 152.50 | 148.25 | 155.75 | 1.68 | 2.15 | 77293 | 77807 | 0 |
| rowchunk_512 | 154.44 | 146.28 | 164.63 | 1.61 | 2.16 | 77293 | 80367 | 0 |

Conclusion: row-chunking is valuable only as a fallback when the fast path cannot
fit.

### 2. Fused Triton full-output prototype

The fused Triton prototype proved the memory direction: tiny scratch is possible.
However, the implementation was not serving-viable yet:

- serving speed was too slow;
- the exactness tests exposed large-vocab/low-temperature drift;
- the implementation should remain research-only until the kernel math and RNG
  behavior are production-grade.

Conclusion: keep the Triton backend hard-gated as experimental; do not use it as
the current serving answer.

### 3. Startup KV-cache capacity claim

We checked whether the memory reduction automatically increased vLLM's startup
KV-cache allocation. It did not move meaningfully at the tested settings. The
sampler transient appears to live in runtime slack rather than in the startup
profiled activation peak that sizes KV cache.

Conclusion: do not claim larger KV cache at default startup. The useful capacity
claim is narrower: high-pressure runtime OOM survival.

## Final high-pressure result

Configuration under test:

```text
VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1
VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto
VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS=2048
VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK=128
--max-num-batched-tokens 4096
--max-num-seqs 16
```

Baseline pressure probe with the same high-pressure serving shape:

| variant | c8 | c12 | c16 | max sampler rows | result |
|---|---:|---:|---:|---:|---|
| baseline | 102.6 tok/s, 0 errors | OOM / 24 errors | OOM / 32 errors | 3072 | fails under pressure |
| rowchunk_256 | 110.7 tok/s, 0 errors | 238.7 tok/s, 0 errors | 216.1 tok/s, 0 errors | 3072 | survives |

After selecting row chunk 128 for the `auto` fallback:

Why `row_chunk=128`: larger chunks had higher normal-path throughput in some runs,
but 384/512 were unsafe under the high-pressure discriminator and 128 provided
the best verified OOM margin among the safe fallback choices. The goal of this
fallback is survival, not peak throughput.

| run | tok/s | errors | mem max |
|---|---:|---:|---:|
| auto c12 | 194.9 | 0 | 80751 MiB |
| auto c16 | 238.8 | 0 | 80751 MiB |

The server log confirmed the intended adaptive switch:

```text
effective_backend=row_chunked
sampler_rows=3072
row_chunk=128
```

Interpretation:

- Under the tested high-pressure shape, 3072 sampler rows crossed the auto
  threshold and used row-chunking.
- The baseline materialized path OOMed at c12/c16.
- The adaptive backend survived c12/c16 with zero request errors.
- Throughput above the threshold should be interpreted as "survives where the
  baseline does not", not as a speedup over a fitting materialized baseline.
- The baseline `mem max` in the pressure table comes from the last surviving
  portion of the run before OOM/server failure. The auto numbers are successful
  c12/c16 runs at the memory ceiling, so the coarse polled MiB values are not
  directly comparable as allocator deltas.

## Correctness and regression coverage

OpenShift/A100 targeted test command:

```text
PYTHONPATH=/tmp/vllm-overlay pytest -q \
  /tmp/vllm-overlay/tests/model_executor/test_diffusion_gemma_fused_lse.py
```

Observed result:

```text
42 passed, 1 xfailed, 20 warnings
```

Fresh targeted rerun of the most relevant tests:

```text
3 passed, 20 warnings in 15.99s
```

The tests cover:

- backend environment parsing and validation;
- the `auto` threshold boundary;
- row-chunked sample/entropy/soft-embedding equivalence against the exact
  materialized cuBLAS/PyTorch path;
- the intentional xfail for the experimental Triton backend exactness gap.

## Current knobs for future work

```text
VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1
VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=eager|auto|cublas_two_pass|row_chunked|triton_full
VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS=2048
VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK=128
VLLM_DIFFUSION_GEMMA_LOG_DECODE_BATCH=1
```

Recommended local serving mode for continued capacity testing:

```text
VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1
VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto
VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK=128
```

Recommended high-pressure discriminator:

```text
--max-num-batched-tokens 4096 --max-num-seqs 16
```

Then compare c8/c12/c16 client concurrency.

## Artifact map

Primary final report artifact outside this worktree:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-auto-backend-final/RESULT.md
```

Supporting artifacts:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-rowchunk-focused-rerun/
.omx/artifacts/diffusion-gemma-sampler-evolve/maxbatched4096-probe-20260616T162937Z/
.omx/artifacts/diffusion-gemma-sampler-evolve/auto-pressure128-20260616T175855Z/
.omx/artifacts/diffusion-gemma-sampler-evolve/DRAW5_CONCEPT_BLEND_SOFTEMBED_PLAN.md
```

File-council review artifacts:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-before-auto-backend-compact/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-after-auto-backend-compact/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-auto-chunk128/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-after-auto128-final/
```

## Bottom line

We discovered that the practical near-term win is **adaptive exact fallback**, not
an always-on fused kernel. The backend should use the normal materialized path
when it fits, because that path is faster. It should switch to exact row-chunking
when sampler rows become large enough that materialization can OOM. This gives a
clear, defensible local result: in the tested high-pressure OpenShift/A100 setup,
`auto` survived c12/c16 where baseline failed.

Future work should focus on a production-grade fused kernel only if it can keep
exactness and serving throughput while reducing the materialized logits floor.
Until then, `auto` + rowchunk_128 is the best verified local baseline for
high-pressure DiffusionGemma OOM-survival experiments.
