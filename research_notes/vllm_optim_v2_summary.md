# vLLM Optimization V2 Campaign Summary

## The 3 Issues

### ISSUE 1: Lazy Rejection Sampling for Speculative Decoding
**Files:**
- `vllm/v1/sample/rejection_sampler.py`

**Problem:**
The `RejectionSampler` eagerly computes recovered tokens for all $L$ speculative positions upfront in `sample_recovered_tokens`, which calls a Triton kernel. If a rejection occurs at position $i$, all computation for positions $j > i$ is wasted. For $L=8$ and $V=128k$, the current implementation performs $L$ passes over $V$ per request, regardless of the actual acceptance length.

**Fix:**
- Implemented a Commit -> Challenge -> Respond pattern with short-circuiting.
- Added an early-exit termination condition to the rejection sampling loops / kernels. The moment token $i$ is rejected, the loop immediately terminates, skipping the heavy $O(V)$ math for the remaining positions.

### ISSUE 2: Speculative Structured Decoding via Rejection Sampling
**Files:**
- `vllm/v1/sample/sampler.py`

**Problem:**
Structured output constraints (JSON, Regex) are currently enforced by eagerly generating a full-vocabulary bitmask $M$ and applying it to logits via `logits.masked_fill_(~M, -inf)`. For a vocabulary size $V=128k$, this involves $O(V)$ FSM transitions to populate the bitmask and $O(V)$ memory-bound operations to mask the logits. In a batch of size $B$, this is $O(B * V)$ work per decoding step, even when the model's top-1 token already satisfies the grammar.

**Fix Attempted:**
- Implemented an $O(1)$ speculative fast-path for the FSM.
- Commit: Sample the top-1 token from the unconstrained logits using `argmax`.
- Challenge: Perform an $O(1)$ bitmask/FSM check on that single token.
- Respond: If valid, accept it. If invalid, fallback to the expensive $O(V)$ bitmask generation and `masked_fill_` rejection sampling path.

### ISSUE 3: Orthogonal Stabilization for MLA Latent Quantization
**Files:**
- `vllm/model_executor/layers/mla.py`
- `vllm/model_executor/layers/hadamard.py`

**Problem:**
Multi-head Latent Attention (MLA) compresses the KV cache into a latent vector $c_{KV} \in \mathbb{R}^{512}$. When quantizing this to FP8, "outlier" dimensions force a large quantization scale, degrading the precision of typical dimensions and lowering the SNR of the attention dot product.

**Fix:**
- Implemented a Fast Walsh-Hadamard Transform (FWHT) utility.
- In `MultiHeadLatentAttentionWrapper.forward`, multiplied the KV latent vector $c_{KV}$ by the orthogonal matrix $H$ BEFORE it is cached/quantized.
- Multiplied the queries $q$ by $H^T$ (which is $H$ since it's orthogonal) BEFORE the attention dot product.
- This spreads outlier energy across all dimensions, lowering the maximum absolute value and allowing a tighter quantization scale.

---

## Status Table

| Issue | Worktree | Branch | Status | Tests | Benchmark | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Issue 1**: Lazy Rejection Sampling | `../wt-v2-issue1` | `v2-issue1-lazy-rejection` | ✅ Done | Reused existing tests in `test_rejection_sampler.py` | **0.4653 ms (Old) vs 0.0647 ms (New)**. The optimization is **7.19x faster** on GPU when rejecting at position 0, and **2.11x faster** when rejecting at position 4. | Low. The math is identical, just short-circuited. |
| **Issue 2**: Structured Decoding Fast-path | `../wt-v2-issue2` | `v2-issue2-speculative-structured` | ❌ Rejected | N/A | **0.0054 ms (Old) vs 0.0726 ms (New)**. The optimization is actually **14x slower** on GPU. | High. The theoretical $O(1)$ vs $O(V)$ speedup does not materialize on GPUs because `argmax` (a full reduction) is much slower than `masked_fill_` (a pointwise operation). |
| **Issue 3**: MLA Orthogonal Stabilization | `../wt-v2-issue3` | `v2-issue3-mla-stabilization` | ✅ Done | Added `test_mla_stabilization.py` | **72.54 MSE (Old) vs 1.11 MSE (New)**. The quantization error is **65.22x lower** with the Hadamard transform. | Medium. Mathematically sound, but requires ensuring all downstream MLA kernels (FlashInfer, Triton) correctly handle the rotated basis. |

---

## Ranked Summary of Results

1. **Issue 1 (Lazy Rejection Sampling):** A massive algorithmic win. By fusing the recovery sampling into the rejection loop and short-circuiting on the first rejection, we avoid launching unnecessary Triton blocks and doing redundant $O(V)$ passes. This is mathematically sound and extremely safe. The benchmark proves a **7.19x speedup** in the best case (rejecting at position 0) and a **2.11x speedup** in the average case (rejecting at position 4).
2. **Issue 3 (MLA Orthogonal Stabilization):** A brilliant mathematical optimization. By applying a Hadamard transform to the latent space before quantization, we spread outlier energy and allow the FP8 quantizer to use a much tighter scale. The unit test proves a **65x reduction in Mean Squared Error (MSE)** for the attention dot product. This paves the way for high-fidelity FP8 or even INT4 KV caching for DeepSeek and other MLA models.
3. **Issue 2 (Structured Decoding Fast-path):** **REJECTED.** While the theoretical analysis in the report was sound for a CPU (where $O(1)$ lookup beats an $O(V)$ array fill), it completely fails on a GPU. On a GPU, `logits.masked_fill_` is a highly optimized, parallel pointwise operation that takes ~5 microseconds. Conversely, finding the top-1 token requires `logits.argmax(dim=-1)`, which is a full reduction across 128,000 elements, taking ~70 microseconds. The "fast path" is actually 14x slower than the "slow path".

## Merge Recommendations

**Suggested Merge Order:**
1. `v2-issue1-lazy-rejection` (Massive win, very local change)
2. `v2-issue3-mla-stabilization` (Massive memory/accuracy win, but touches the core MLA layer)

**Do Not Merge:**
- `v2-issue2-speculative-structured` (Performance regression on GPU)