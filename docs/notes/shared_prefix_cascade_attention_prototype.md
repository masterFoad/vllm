# Shared-Prefix Cascade Attention Prototype (FlashInfer, experimental)

## Goal

Prototype a bounded, benchmarkable version of shared-prefix decode optimization in vLLM by enabling the existing dormant FlashInfer cascade-attention path behind an explicit experimental flag.

This is **not** a new kernel implementation. It is a first-step integration prototype that tries to turn an existing but disabled path into an opt-in feature that can be evaluated safely on shared-prefix workloads.

## Why this direction

vLLM already has:

- Automatic Prefix Caching (APC)
- common-prefix detection in the V1 GPU runner
- a mature FlashAttention cascade path
- a partially wired FlashInfer cascade metadata/forward path that is currently hard-disabled

That suggests a low-to-medium scope prototype is possible without introducing a brand new algorithmic backend.

## Hypothesis

When many decode requests share a long common prefix, the FlashInfer cascade path should reduce redundant KV reads versus standard per-request decode, improving throughput on shared-prefix workloads.

## Bounded prototype scope

1. Add an explicit opt-in environment flag for FlashInfer cascade attention.
2. Keep the current default behavior unchanged.
3. Reuse existing common-prefix detection and metadata-building logic.
4. Enable the dormant FlashInfer cascade path only when the flag is on and existing guardrails pass.
5. Add a benchmark mode focused on shared-prefix traffic rather than generic prompts.

## Non-goals

- no new CUDA/Triton kernel
- no claims of universal speedup
- no change to non-FlashInfer backends
- no attempt to support all incompatible features at once

## Key risks

- the existing FlashInfer cascade path may have correctness or stability issues, which is likely why it was disabled
- generic benchmark workloads may show no win because they do not exercise shared prefixes
- numerical drift may appear relative to non-cascade execution

## Success criteria

- feature remains opt-in
- default behavior is unchanged
- shared-prefix benchmark runs on OpenShift
- branch-vs-main comparison completes
- outputs remain acceptably stable on the benchmarked workload
- performance delta is measured, even if negative

## Likely files

- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/envs.py`
- `vllm/benchmarks/throughput.py` or OpenShift benchmark script path
- `examples/yamls/vllm_bench_h2d_pr.yaml`

## Benchmark shape

The existing OpenShift PR benchmark uses generic prompts, which are unlikely to stress shared-prefix decode.

For this prototype, the benchmark should add a shared-prefix case such as:

- one long common system prefix
- many distinct suffix prompts
- deterministic sampling
- throughput and token equality checks against baseline

## Interpretation rule

A null or negative result is still useful if correctness holds and the benchmark is reproducible. That tells us the dormant path either needs deeper work or is not yet worth enabling broadly.
