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
number of sampler rows exceeds a configured threshold chosen as an OOM-risk
proxy for the tested high-pressure shape.

The headline evidence from OpenShift/A100 testing:

- Normal low-pressure serving: always-on row-chunking is slower than baseline,
  so it should not be enabled unconditionally.
- High-pressure serving with `--max-num-batched-tokens 4096 --max-num-seqs 16`:
  the baseline materialized path OOMed at client concurrency 12 and 16, while
  the adaptive backend completed both runs with zero request errors.
- ShareGPT-style serving validation (`vllm bench serve`, first 50 prompts,
  output length 128) reproduced the same capacity behavior: baseline completed
  c=3, partially failed at c=12, and fully failed at c=16; adaptive completed
  c=3/c=12/c=16 with zero request errors.
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
- It is not a lower-latency/per-token speedup when both baseline and adaptive
  fit. It can still be a useful aggregate throughput increase at pressure points
  where baseline returns errors or dies and adaptive continues serving requests.
- It does not increase vLLM's startup KV-cache allocation at the tested default
  settings.
- It does not make the experimental Triton fused backend PR-ready.
- It does not address the separate logprobs-on bug class discussed during the
  investigation.
- It does not characterize tensor-parallel (`TP>1`) behavior. All reported
  OpenShift results here are single-GPU/TP=1; row-chunking could introduce
  different communication overheads in multi-GPU tensor-parallel deployments.

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
- Throughput in failed runs is tricky: `request_throughput` and total token/s can
  be inflated by fast failures or prompt-token accounting. For capacity claims,
  prioritize completed requests, non-empty errors, output token/s from successful
  requests, and server OOM evidence.
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
uses `eager`, not `auto`. The current `auto` policy is a static
`sampler_rows` threshold; it does not query allocator headroom or predict OOM
dynamically. The `2048` threshold and `128` fallback chunk were selected for the
tested A100/OpenShift shape and should be retuned before treating them as
universal defaults; code comments and PR text should keep that hardware-specific
warning visible.

## Paths we tried and rejected

### 1. Always-on row-chunking

Always-on row-chunking lowers peak materialization pressure, but it regresses the
normal serving path. That makes it unsuitable as the default.

OpenShift/A100 row-chunk focused c=3 rerun:

| variant      | mean tok/s | min tok/s | max tok/s | mean p50 s | mean p95 s | ready MiB | max bench MiB | errors |
| ------------ | ---------: | --------: | --------: | ---------: | ---------: | --------: | ------------: | -----: |
| baseline     |     224.70 |    196.33 |    245.35 |       1.07 |       1.57 |     77293 |         77295 |      0 |
| rowchunk_128 |     138.78 |    135.39 |    141.81 |       1.70 |       2.63 |     77293 |         77295 |      0 |
| rowchunk_256 |     152.50 |    148.25 |    155.75 |       1.68 |       2.15 |     77293 |         77807 |      0 |
| rowchunk_512 |     154.44 |    146.28 |    164.63 |       1.61 |       2.16 |     77293 |         80367 |      0 |

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

Preliminary pressure probe with the same high-pressure serving shape, before selecting the final `row_chunk=128` fallback:

| variant      |                    c8 |                   c12 |                   c16 | max sampler rows | result               |
| ------------ | --------------------: | --------------------: | --------------------: | ---------------: | -------------------- |
| baseline     | 102.6 tok/s, 0 errors |       OOM / 24 errors |       OOM / 32 errors |    3072 observed | fails under pressure |
| rowchunk_256 | 110.7 tok/s, 0 errors | 238.7 tok/s, 0 errors | 216.1 tok/s, 0 errors |    3072 observed | survives             |

After selecting row chunk 128 for the `auto` fallback:

Why `row_chunk=128`: larger chunks had higher normal-path throughput in some runs,
but 384/512 were unsafe under the high-pressure discriminator and 128 provided
the best verified OOM margin among the safe fallback choices. The goal of this
fallback is survival, not peak throughput.

| run      | tok/s | errors |   mem max |
| -------- | ----: | -----: | --------: |
| auto c12 | 194.9 |      0 | 80751 MiB |
| auto c16 | 238.8 |      0 | 80751 MiB |

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

## ShareGPT first-50 A/B validation

To check that the pressure result was not only an artificial smoke prompt, we
also ran a standard vLLM benchmark shape on OpenShift using `vllm bench serve`
with a ShareGPT slice.

Configuration:

```text
dataset: ShareGPT first 50 prompts from a first-500 valid-record slice
backend: openai-chat
endpoint: /v1/chat/completions
sharegpt-output-len: 128
request-rate: inf
seed: 123
server: google/diffusiongemma-26B-A4B-it
server flags: --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192
              --max-num-seqs 16 --max-num-batched-tokens 4096 --trust-remote-code
```

Result artifact:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/sharegpt-ab-20260616T194105Z/
```

Results:

| variant | c | completed | non-empty errors | output tok/s | req/s | total tok/s | p95 TTFT ms | server max MiB | failure evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 3 | 50/50 | 0 | 183.89 | 1.58 | 466.49 | 3373 | 80991 | none in this fanout |
| baseline | 12 | 12/50 | 38 | 0.00 | 16.68 | 1736.00 | n/a | 80991 | 500s / CUDA OOM evidence |
| baseline | 16 | 0/50 | 50 | 0.00 | 0.00 | 0.00 | n/a | 80991 | engine dead / connection errors |
| adaptive | 3 | 50/50 | 0 | 175.21 | 1.51 | 444.75 | 3750 | 80751 | none |
| adaptive | 12 | 50/50 | 0 | 152.68 | 1.31 | 386.62 | 12114 | 80751 | none |
| adaptive | 16 | 50/50 | 0 | 160.70 | 1.38 | 407.31 | 15809 | 80751 | none |

Interpretation:

- The adaptive backend reproduced the high-pressure capacity result on a
  ShareGPT-style workload: baseline completed the low-pressure c=3 run,
  partially failed at c=12, and fully failed at c=16; adaptive completed all
  three fanouts with zero request errors.
- Baseline c=12 should be treated as a failed run, not a throughput datapoint:
  only 12 of 50 requests completed, 38 returned `Internal Server Error`, and
  reported output token/s was 0. The high `total tok/s` value is not a useful
  serving capacity metric because it includes fast failed/prompt-token
  accounting; the `0.00` output token/s is also a benchmark artifact of the
  heavily errored run, so the reliable conclusion is failure, not speed.
- Adaptive c=12/c=16 aggregate output token/s is lower than baseline c=3, but
  that is still usable output throughput under a fanout where baseline is not
  producing reliable output and/or the engine dies.
- Adaptive survival at c=12/c=16 comes with much higher p95 TTFT
  (~12.1-15.8s versus ~3.4-3.8s at c=3). This is an OOM-survival/capacity
  result, not proof of good interactive latency under high load.
- At c=3, adaptive was about 5% lower in output token/s in this single run
  (175.21 vs 183.89). Because the `auto` policy should remain on the
  materialized path below the row threshold, this difference is likely Python
  branching overhead, decode-batch logging overhead, or standard runtime
  variance. A repeated low-pressure run without decode-batch logging is required
  to prove that definitively. Disable
  `VLLM_DIFFUSION_GEMMA_LOG_DECODE_BATCH=1` for clean performance runs.
- This benchmark used output length 128. That is intentionally modest; the fact
  that baseline still fails at c=12/c=16 is a strong pressure signal. Longer
  ShareGPT soaks are still recommended before upstream PR claims.
- The ShareGPT data above is one run per fanout; it is strong enough for a local
  capacity discriminator, but repeated runs are needed before treating the exact
  throughput/latency values as stable.

The adaptive server log confirmed the intended switch:

```text
DiffusionGemma streamed sampler path is active
backend=auto
effective_backend=row_chunked
sampler_rows=3072
auto_max_materialized_rows=2048
row_chunk=128
```

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
test_diffusion_gemma_streamed_backend_validation
test_diffusion_gemma_streamed_auto_backend_boundary
test_diffusion_gemma_row_chunked_sample_soft_embeds_matches_cublas

3 passed, 20 warnings in 15.99s
```

The tests cover:

- backend environment parsing and validation;
- the `auto` threshold boundary;
- row-chunked sample/entropy/soft-embedding equivalence against the exact
  materialized cuBLAS/PyTorch path;
- the intentional xfail for the experimental Triton backend exactness gap.

Important PR-readiness gap: these tests cover the helper math and backend
selection, but they are not a full model-execution integration test of
`_compiled_sample_step_from_streamed` through vLLM's generation state machine.
Before an upstream PR, add or parameterize an end-to-end DiffusionGemma
generation test that sets `VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1` and
`VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto`, then verifies the streamed sampler
integrates correctly with canvas history, sampler output state, and normal
request completion.

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

Primary final report artifact outside this worktree. These `.omx` paths are local/private workspace evidence, not upstream vLLM documentation assets:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-auto-backend-final/RESULT.md
```

Supporting artifacts:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-rowchunk-focused-rerun/
.omx/artifacts/diffusion-gemma-sampler-evolve/maxbatched4096-probe-20260616T162937Z/
.omx/artifacts/diffusion-gemma-sampler-evolve/auto-pressure128-20260616T175855Z/
.omx/artifacts/diffusion-gemma-sampler-evolve/sharegpt-ab-20260616T194105Z/
.omx/artifacts/diffusion-gemma-sampler-evolve/DRAW5_CONCEPT_BLEND_SOFTEMBED_PLAN.md
```

File-council review artifacts:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-before-auto-backend-compact/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-after-auto-backend-compact/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-auto-chunk128/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-after-auto128-final/
.omx/artifacts/diffusion-gemma-sampler-evolve/file-council-sharegpt-report-20260616T195652Z/
.omx/artifacts/diffusion-gemma-sampler-evolve/file-council-sharegpt-report-final-20260616T200151Z/
```

## Handoff: how to continue from this branch

This section is meant for a future session starting cold. The current branch is
not a PR branch yet; treat it as the local baseline for continued DiffusionGemma
capacity work.

### Branch and file map

Branch name used when written:

```text
branch: masterfoad/diffusion-gemma-adaptive-sampler-baseline-20260616
fork remote used when written: git@github.com:masterFoad/vllm.git
upstream base when written: vLLM origin/main 8e27a9c21
```

Main files to read first:

```text
docs/design/diffusion_gemma_adaptive_sampler_report.md
vllm/model_executor/models/diffusion_gemma.py
vllm/model_executor/models/diffusion_gemma_fused_lse.py
tests/model_executor/test_diffusion_gemma_fused_lse.py
```

Start by checking whether the branch is still based on current upstream.
Remote names can differ by checkout; use whichever remote points at
`vllm-project/vllm`:

```bash
git remote -v
# If needed:
git remote add upstream https://github.com/vllm-project/vllm.git || true
git fetch upstream main
git status --short --branch
git log --oneline upstream/main..HEAD
```

If continuing experiments rather than preparing a PR, keep changes local or push
to the `masterFoad/vllm` fork. Do not open a PR until the broader validation and
vLLM contribution checks are done.

### Skills and review surfaces used

We used three local reasoning/review surfaces. They are part of the process
history, not runtime dependencies. The paths below are local/private workflow
conventions from this workstation; sanitize or remove them before any upstream
PR-facing documentation.

#### File Council

Local skill source: Codex skill registry / file-council skill. The exact local path is machine-specific and not a vLLM dependency.

What it does: sends selected local files plus one prompt to ChatGPT, Claude,
Gemini Pro, and Gemini Flash through the configured IBM LiteLLM REST gateway. It
writes durable JSONL/raw artifacts, but it does not synthesize or arbitrate by
itself.

When we used it:

- before/after significant design changes;
- after the adaptive backend/report was written;
- after the report-review pass caught a real documentation bug: the report
  initially forgot that `VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1` is required
  before `VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto` has any effect.

Reusable command shape:

```bash
python "$FILE_COUNCIL_SKILL_DIR/scripts/run_file_council.py" \
  --session "dg-adaptive-review-$(date -u +%Y%m%dT%H%M%SZ)" \
  --state-dir .omx/file-council \
  --prompt "Review these files for overclaims, correctness risks, and missing caveats." \
  --file /absolute/path/to/docs/design/diffusion_gemma_adaptive_sampler_report.md \
  --file /absolute/path/to/vllm/model_executor/models/diffusion_gemma.py \
  --file /absolute/path/to/vllm/model_executor/models/diffusion_gemma_fused_lse.py \
  --file /absolute/path/to/tests/model_executor/test_diffusion_gemma_fused_lse.py \
  --targets chatgpt claude gemini gemini-flash \
  --thinking high \
  --no-web-search
```

Most useful local/private file-council artifacts from this investigation:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-before-auto-backend-compact/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-after-auto-backend-compact/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-auto-chunk128/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-file-council-after-auto128-final/
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-report-commit-review*/
```

#### Draw5 Operator Cards

Local skill source: Draw5 Operator Cards skill. The exact local path is machine-specific and not a vLLM dependency.

What it did for this work: forced the optimization search into an operator
ladder instead of chasing one shiny kernel. The useful operators were:

- proxy metric -> true objective alignment: do not trust scratch-only wins;
  require serving tok/s, latency, errors/OOM, and exactness;
- flat retrieved hand -> typed operator ladder: separate primitives, process,
  risk, and validation operators;
- GPU kernel/SIMD/fusion analogies: useful for fused-kernel hypotheses, but only
  after exactness and serving gates.

Commands used/usable:

```bash
SKILL_DIR="$DRAW5_SKILL_DIR"
python "$SKILL_DIR/scripts/operator_retriever.py" build \
  --cards "$SKILL_DIR/cards" \
  --index .draw5_operator_index \
  --provider hashing \
  --backend numpy

python "$SKILL_DIR/scripts/operator_retriever.py" query \
  --index .draw5_operator_index \
  --query "DiffusionGemma sampler memory OOM survival GPU kernel row chunking adaptive fallback soft embedding exactness benchmark strategy" \
  --mode deck-of-decks \
  --top 10 \
  --format prompt
```

Latest local/private query artifact:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/20260616T-report-handoff-skills/draw5-query.md
```

#### Concept Blending

Local skill source: Concept Blending skill. The exact local path is machine-specific and not a vLLM dependency.

What it did for this work: generated falsifiable candidate mechanisms and then
clamped them against prior art, mechanism, constraints, and strongest objection.
The strongest surviving concepts were:

1. **Exact Row-Chunk Autotune** — choose row-chunking by a Pareto frontier and
   use it only as a fallback. This is the idea that became the practical
   `auto` backend baseline.
2. **Flash-GEMM SoftEmbed** — future fused/tiled kernel direction, but only if it
   keeps exactness and throughput after a hard isolated gate.
3. **Liger/CCE Borrowed Skeleton** — use fused linear cross-entropy / Liger-style
   tiling as inspiration, not as a new dependency.

Primary local plan artifact:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/DRAW5_CONCEPT_BLEND_SOFTEMBED_PLAN.md
```

### OpenShift operating recipe

Use OpenShift for all GPU claims. Do not claim GPU correctness/performance from
WSL-only pytest or CPU checks.

Project and model:

```text
OpenShift project: wdu-research
Model: google/diffusiongemma-26B-A4B-it
Working image: vllm/vllm-openai:gemma
PVC: model-cache-rwx mounted at /model-cache
HF cache: /model-cache/huggingface
```

Login should use a fresh token from the user/environment; do not write tokens
into repo artifacts:

```bash
oc login --token=<fresh-token> --server=<openshift-api-server>
oc project wdu-research
```

Before creating anything, check whether a server pod already exists:

```bash
oc get appwrappers,jobs,pods -l app=diffusiongemma-chat -o wide
```

If reusing a pod, first verify that its image/command/env match the experiment;
an old running pod may be from a different branch or backend:

```bash
oc describe pod/$POD | grep -E 'Image:|VLLM_DIFFUSION_GEMMA|max-num-batched|max-num-seqs' || true
oc logs pod/$POD | grep -E 'VLLM_DIFFUSION_GEMMA|effective_backend|sampler_rows|row_chunk' | tail -50 || true
```

If a Running pod exists, reuse it and port-forward rather than creating another
job:

```bash
POD=$(oc get pods -l app=diffusiongemma-chat \
  -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' \
  | head -n 1)
if [ -z "$POD" ]; then echo "No running pod found; create/reuse a job first"; exit 1; fi
oc port-forward pod/$POD 8000:8000
```

Smoke checks:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq .

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/diffusiongemma-26B-A4B-it",
    "messages": [{"role": "user", "content": "Write two sentences about diffusion language models."}],
    "max_tokens": 128,
    "temperature": 0.7
  }' | jq .
```

Known local helpers from the parent repo:

```bash
scripts/start_diffusiongemma_port_forward.sh --status
JOB=<new-job-name> scripts/start_diffusiongemma_port_forward.sh
python3 scripts/diffusiongemma_chat_ui.py --open
python3 scripts/diffusiongemma_canvas_ui.py --open
```

The `vllm serve` command below is the container entrypoint shape for the
OpenShift AppWrapper/Job, not a command to run on the WSL host. If no compatible
pod exists, create a new AppWrapper/Job and inject the adaptive sampler env vars
into the container environment before `vllm serve`. Known-good parent-repo
templates to inspect before recreating the job:

```text
.omx/artifacts/diffusiongemma-openshift/tavily-20260611t145959/appwrapper.yaml
.omx/artifacts/diffusiongemma-openshift/20260611T103748Z/run_diffusiongemma_server.sh
.omx/artifacts/diffusiongemma-openshift/20260611T103748Z/README-talk-to-diffusiongemma.md
```

Serving command baseline inside the pod/job:

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code
```

For adaptive sampler experiments, add:

```bash
export VLLM_DIFFUSION_GEMMA_STREAMED_SAMPLER=1
export VLLM_DIFFUSION_GEMMA_STREAMED_BACKEND=auto
export VLLM_DIFFUSION_GEMMA_STREAMED_AUTO_MAX_MATERIALIZED_ROWS=2048
export VLLM_DIFFUSION_GEMMA_STREAMED_ROW_CHUNK=128
# Optional diagnostics:
export VLLM_DIFFUSION_GEMMA_LOG_DECODE_BATCH=1
```

Pressure discriminator used for the strongest result:

```text
--max-num-batched-tokens 4096
--max-num-seqs 16
client concurrency sweep: c8, c12, c16
```

Benchmark harness notes:

```text
c8:  concurrency=8,  total requests=16
c12: concurrency=12, total requests=24
c16: concurrency=16, total requests=32
max_tokens: 128 in the pressure runs
metrics: successful completion tokens / wall seconds, request errors, p50 when captured, and coarse GPU memory polling
```

ShareGPT harness shape used for the normal dataset validation:

```bash
python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path /tmp/sharegpt-bench/sharegpt_first500.json \
  --disable-shuffle \
  --num-prompts 50 \
  --sharegpt-output-len 128 \
  --request-rate inf \
  --max-concurrency <3|12|16> \
  --seed 123 \
  --save-result \
  --save-detailed
```

The final pressure run artifacts are the source of truth for exact fields:

```text
.omx/artifacts/diffusion-gemma-sampler-evolve/auto-pressure128-20260616T175855Z/auto_pressure128/bench_c12.json
.omx/artifacts/diffusion-gemma-sampler-evolve/auto-pressure128-20260616T175855Z/auto_pressure128/bench_c16.json
.omx/artifacts/diffusion-gemma-sampler-evolve/auto-pressure128-20260616T175855Z/auto_pressure128/server.log
.omx/artifacts/diffusion-gemma-sampler-evolve/auto-pressure128-20260616T175855Z/auto_pressure128/key.log
```

For a future rerun, save the server log, client JSON, pod description, and GPU
memory poll for every variant. For OOM cases, preserve the CUDA OOM traceback or
pod termination reason. With `VLLM_DIFFUSION_GEMMA_LOG_DECODE_BATCH=1`, verify
that logs include `effective_backend=row_chunked`, `sampler_rows`, and
`row_chunk=128` once the threshold is crossed.

### Testing strategy used

The testing pyramid that worked best:

1. **Static/local sanity**
   - `py_compile` changed Python files;
   - `git diff --check` for whitespace/conflict artifacts;
   - these are preflight checks only, not GPU sampler validation.
2. **OpenShift GPU unit/correctness**
   - run `tests/model_executor/test_diffusion_gemma_fused_lse.py` in the GPU
     image/overlay;
   - require row-chunked equivalence against the materialized reference within
     the tolerances documented in the tests; "exact" means same algorithmic
     reference path/distribution, not bitwise identity unless the test explicitly
     says so;
   - keep the Triton exactness drift as an intentional xfail, not a hidden pass.
3. **Serving smoke**
   - `/v1/models` and a small `/v1/chat/completions` call;
   - keep client fanout low for normal smoke because the server has died under
     excessive fanout.
4. **Normal-path throughput**
   - c=3 repeated rounds to catch regressions when memory pressure is not the
     binding constraint;
   - this is where always-on row-chunking failed the no-regression bar.
5. **High-pressure OOM/capacity discriminator**
   - raise `max_num_batched_tokens` / `max_num_seqs` and compare c8/c12/c16;
   - success criterion is not speedup; it is baseline OOM vs adaptive zero
     errors at the same shape.
6. **ShareGPT-style serving validation**
   - run `vllm bench serve --dataset-name sharegpt` against both variants;
   - count non-empty errors, not just process return code;
   - interpret failed baseline fanouts as capacity failures, not comparable
     throughput datapoints.
7. **Review gates**
   - file-council before/after significant pivots;
   - explicitly separate evidence from inference and document non-claims.

Useful commands after code edits:

```bash
.venv/bin/python -m py_compile \
  vllm/model_executor/models/diffusion_gemma.py \
  vllm/model_executor/models/diffusion_gemma_fused_lse.py \
  tests/model_executor/test_diffusion_gemma_fused_lse.py

git diff --check -- \
  vllm/model_executor/models/diffusion_gemma.py \
  vllm/model_executor/models/diffusion_gemma_fused_lse.py \
  tests/model_executor/test_diffusion_gemma_fused_lse.py \
  docs/design/diffusion_gemma_adaptive_sampler_report.md
```

Useful OpenShift pytest shape:

```bash
PYTHONPATH=/tmp/vllm-overlay pytest -q \
  /tmp/vllm-overlay/tests/model_executor/test_diffusion_gemma_fused_lse.py
```

### Failed experiments and lessons

1. **Commit-row skipping**
   - Finding: commit rows do waste sampler work.
   - Why it was not the main win: commit rows are rare and skipping them inside a
     compiled whole-batch region does not remove the main full-vocab memory peak
     without dynamic partitioning.
   - Lesson: optimize the denoise path that every row pays, not the rare commit
     path.

2. **Sampler transient reduction / chunked softmax moments**
   - Finding: it correctly reduced redundant sampler scratch and exposed the
     right math: entropy from `logsumexp - E[x]`, and soft embedding as
     `E_p[embedding]`.
   - Bugs found/fixed: bf16 cross-chunk accumulation was lossy; fp32 multiply was
     too slow under vLLM's TF32-disabled serving runtime. The right compromise was
     bf16 tensor-core multiply with fp32 cross-chunk accumulation.
   - Lesson: microbench speed can lie if runtime matmul precision differs from
     serving.

3. **KV-cache headroom claim**
   - Finding: startup KV cache did not move meaningfully between baseline and
     patched runs.
   - Lesson: sampler scratch lived in runtime slack; do not claim larger startup
     KV allocation.

4. **Always-on row-chunking**
   - Finding: it survived pressure but was slower in normal c=3 serving.
   - Lesson: row-chunking is an OOM fallback, not a default speed path.

5. **Rowchunk 384/512**
   - Finding: larger chunks looked attractive for throughput in some normal runs,
     but failed the high-pressure discriminator.
   - Lesson: select fallback chunk by survival margin first; `128` is the current
     verified safe default for `auto`.

6. **Triton fused full/single-pass prototypes**
   - Finding: tiny scratch and promising isolated pieces, but not serving-ready:
     slow live serving and exactness drift in hard large-vocab/low-temperature
     cases.
   - Lesson: keep `triton_full` gated as experimental. Future fused kernels need
     exactness parity and serving throughput gates before integration.

7. **Low-fanout throughput-only tests**
   - Finding: they can show regressions but cannot prove capacity value.
   - Lesson: combine low-pressure throughput with high-pressure OOM survival;
     neither alone is sufficient.

### Where to go next

Recommended next work, in order:

1. **Longer soak of the current adaptive baseline**
   - same high-pressure shape;
   - more prompts and longer wall time;
   - include a repeated ShareGPT run at c12/c16 with 100-200 prompts;
   - optionally increase output length after the first longer soak. The current
     output length 128 is enough to demonstrate that baseline fails under this
     concurrency shape, but longer generations would better characterize latency
     and stability.
   - confirm zero errors and stable memory near c12/c16.
2. **Broader shape sweep**
   - `gpu_memory_utilization` variants;
   - `max_num_batched_tokens` variants;
   - `max_num_seqs` variants;
   - possibly smaller GPUs if available.
   - identify the highest adaptive concurrency / batch shape that still produces
     stable output, and report it separately from per-token speed.
3. **CI-runnable unit cleanup**
   - make sure exact helper tests can run in upstream vLLM CI without the custom
     OpenShift overlay when possible.
4. **Future fused kernel only behind a hard gate**
   - do not revive the current Triton prototype as serving code;
   - de-risk a production kernel in isolation first;
   - require exactness in low-temp/large-vocab cases, throughput parity, and a
     demonstrated capacity benefit before wiring it into serving.
5. **Separate logprobs work**
   - keep the known logprobs-on bug/failure class separate from this memory PR
     line unless the user explicitly asks to merge the efforts.

Before any upstream PR, add a PR-readiness checklist pass:

- rebase on current upstream vLLM;
- run the relevant vLLM lint/test subset in an upstream-compatible environment;
- confirm env-var naming and feature gating are acceptable;
- remove or move private OpenShift/tooling handoff details out of PR-facing docs;
  this report is intentionally useful as a local handoff, but should not be
  copied into an upstream PR as-is because it contains workstation-specific
  paths, OpenShift operational details, and local skill-process history.
- search open vLLM PRs/issues for duplicate DiffusionGemma sampler memory work;
- add at least one end-to-end generation integration test for the opt-in
  streamed `auto` path, not only helper-level numerical tests;
- write the PR as an opt-in OOM mitigation with test evidence and AI-assistance
  disclosure, not as a general speedup.

Best one-line continuation prompt for a new session:

```text
Read docs/design/diffusion_gemma_adaptive_sampler_report.md on branch
masterfoad/diffusion-gemma-adaptive-sampler-baseline-20260616 and continue
OpenShift validation of the opt-in DiffusionGemma adaptive sampler baseline using
STREAMED_SAMPLER=1, BACKEND=auto, ROW_CHUNK=128, and the
max-num-batched-tokens=4096/max-num-seqs=16 pressure shape.
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
