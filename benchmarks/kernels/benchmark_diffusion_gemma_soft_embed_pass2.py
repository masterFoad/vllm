# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DiffusionGemma pass-2 soft-embed recompute.

This de-risks only the second pass of a fused sampler: callers provide LSE and
we compute softmax(softcap(hidden @ lm_head.T) / temp) @ embedding without
materializing full probabilities. It intentionally excludes pass-1 stats,
Gumbel, tensor-parallel merge, and serving integration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from vllm.model_executor.models.diffusion_gemma_fused_lse import (
    diffusion_gemma_softcap_row_chunked_soft_embed_from_lse,
    diffusion_gemma_softcap_soft_embed_from_lse,
)


@dataclass
class BenchResult:
    name: str
    ms_mean: float
    ms_min: float
    peak_mib: float
    peak_reserved_mib: float


@torch.inference_mode()
def _materialized_full(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
) -> torch.Tensor:
    scores = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    scores.div_(softcap).tanh_().mul_(softcap)
    scores.div_(temperature[:, None].clamp(min=1.0e-10))
    probs = scores.softmax(dim=-1)
    return torch.mm(probs.to(embed_weight.dtype), embed_weight,
                    out_dtype=torch.float32)


@torch.inference_mode()
def _materialized_from_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    lse: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
) -> torch.Tensor:
    scores = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    scores.div_(softcap).tanh_().mul_(softcap)
    scores.div_(temperature[:, None].clamp(min=1.0e-10))
    scores.sub_(lse[:, None]).exp_()
    return torch.mm(scores.to(embed_weight.dtype), embed_weight,
                    out_dtype=torch.float32)


@torch.inference_mode()
def _fp32_oracle_from_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    embed_weight: torch.Tensor,
    lse: torch.Tensor,
    softcap: float,
    temperature: torch.Tensor,
) -> torch.Tensor:
    scores = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    scores.div_(softcap).tanh_().mul_(softcap)
    scores.div_(temperature[:, None].clamp(min=1.0e-10))
    scores.sub_(lse[:, None]).exp_()
    return scores @ embed_weight.float()


@torch.inference_mode()
def _measure(
    name: str,
    fn: Callable[[], torch.Tensor],
    warmup: int,
    repeat: int,
) -> tuple[BenchResult, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    del out
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()

    times = []
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    peak_reserved = torch.cuda.max_memory_reserved() - base_reserved
    for _ in range(repeat):
        del out
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        peak = max(peak, torch.cuda.max_memory_allocated() - base)
        peak_reserved = max(
            peak_reserved, torch.cuda.max_memory_reserved() - base_reserved
        )

    return (
        BenchResult(
            name=name,
            ms_mean=float(sum(times) / len(times)),
            ms_min=float(min(times)),
            peak_mib=float(peak / 2**20),
            peak_reserved_mib=float(peak_reserved / 2**20),
        ),
        out,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--embed-size", type=int, default=None)
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--block-e", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--row-chunks", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--max-acceptable-ratio", type=float, default=1.15)
    parser.add_argument(
        "--min-peak-reduction",
        type=float,
        default=0.20,
        help=(
            "Required allocated-peak reduction vs materialized_from_lse "
            "for proceed."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--skip-triton-pass2", action="store_true")
    parser.add_argument("--skip-fp32-oracle", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.manual_seed(args.seed)
    embed_size = args.embed_size or args.hidden_size
    device = "cuda"

    hidden = torch.randn(
        args.rows, args.hidden_size, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(
        args.vocab_size, args.hidden_size, device=device, dtype=torch.bfloat16
    ) / args.hidden_size**0.5
    embed_weight = torch.randn(
        args.vocab_size, embed_size, device=device, dtype=torch.bfloat16
    ) / embed_size**0.5
    temperature = torch.full(
        (args.rows,), args.temperature, device=device, dtype=torch.float32
    )

    with torch.inference_mode():
        scores = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        scores.div_(args.softcap).tanh_().mul_(args.softcap)
        scores.div_(temperature[:, None])
        lse = scores.logsumexp(dim=-1)
        del scores
        torch.cuda.synchronize()

    full_res, full = _measure(
        "materialized_full",
        lambda: _materialized_full(
            hidden, weight, embed_weight, args.softcap, temperature
        ),
        args.warmup,
        args.repeat,
    )
    lse_res, lse_ref = _measure(
        "materialized_from_lse",
        lambda: _materialized_from_lse(
            hidden, weight, embed_weight, lse, args.softcap, temperature
        ),
        args.warmup,
        args.repeat,
    )
    if args.skip_fp32_oracle:
        oracle_res = None
        oracle = None
    else:
        oracle_res, oracle = _measure(
            "fp32_oracle_from_lse",
            lambda: _fp32_oracle_from_lse(
                hidden, weight, embed_weight, lse, args.softcap, temperature
            ),
            args.warmup,
            args.repeat,
        )
    if args.skip_triton_pass2:
        pass2_res = None
        pass2 = None
    else:
        pass2_res, pass2 = _measure(
            "triton_pass2_from_lse",
            lambda: diffusion_gemma_softcap_soft_embed_from_lse(
                hidden,
                weight,
                embed_weight,
                lse,
                args.softcap,
                temperature,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                block_e=args.block_e,
                num_warps=args.num_warps,
            ),
            args.warmup,
            args.repeat,
        )

    row_chunk_results = []
    best_row_chunk = None
    for row_chunk in args.row_chunks:
        row_res, row_out = _measure(
            f"row_chunked_from_lse_{row_chunk}",
            lambda row_chunk=row_chunk: (
                diffusion_gemma_softcap_row_chunked_soft_embed_from_lse(
                    hidden,
                    weight,
                    embed_weight,
                    lse,
                    args.softcap,
                    temperature,
                    row_chunk_size=row_chunk,
                )
            ),
            args.warmup,
            args.repeat,
        )
        row_payload = {
            **asdict(row_res),
            "row_chunk_size": row_chunk,
            "max_abs_vs_materialized_full": float(
                (row_out - full).abs().max().item()
            ),
            "max_abs_vs_materialized_from_lse": float(
                (row_out - lse_ref).abs().max().item()
            ),
            "max_abs_vs_fp32_oracle": (
                float((row_out - oracle).abs().max().item())
                if oracle is not None else None
            ),
            "ratio_vs_materialized_full_ms": row_res.ms_mean / full_res.ms_mean,
            "ratio_vs_materialized_from_lse_ms": (
                row_res.ms_mean / lse_res.ms_mean
            ),
            "ratio_vs_materialized_full_peak": (
                row_res.peak_mib / full_res.peak_mib
            ),
            "ratio_vs_materialized_from_lse_peak": (
                row_res.peak_mib / lse_res.peak_mib
            ),
            "estimated_score_prob_scratch_mib": (
                row_chunk * args.vocab_size * (4 + 2) / 2**20
            ),
        }
        row_chunk_results.append(row_payload)
        if best_row_chunk is None or row_res.ms_mean < best_row_chunk[
                "ms_mean"]:
            best_row_chunk = row_payload

    viable_row_chunks = [
        r for r in row_chunk_results
        if r["ratio_vs_materialized_from_lse_ms"] <= args.max_acceptable_ratio
        and r["ratio_vs_materialized_from_lse_peak"]
        <= 1.0 - args.min_peak_reduction
    ]
    selected_row_chunk = (
        min(viable_row_chunks, key=lambda r: r["row_chunk_size"])
        if viable_row_chunks else None
    )

    max_abs_vs_full = (
        float((pass2 - full).abs().max().item())
        if pass2 is not None else None
    )
    max_abs_vs_lse = (
        float((pass2 - lse_ref).abs().max().item())
        if pass2 is not None else None
    )
    ratio_vs_full = (
        pass2_res.ms_mean / full_res.ms_mean if pass2_res is not None else None
    )
    ratio_vs_lse = (
        pass2_res.ms_mean / lse_res.ms_mean if pass2_res is not None else None
    )
    best_row_ratio = (
        best_row_chunk["ratio_vs_materialized_from_lse_ms"]
        if best_row_chunk is not None else float("inf")
    )
    selected_row_ratio = (
        selected_row_chunk["ratio_vs_materialized_from_lse_ms"]
        if selected_row_chunk is not None else float("inf")
    )
    decision = (
        "proceed"
        if selected_row_chunk is not None
        else "try_l2_scratch" if best_row_ratio <= 1.30 else "not_serving_ready"
    )
    payload = {
        "shape": {
            "rows": args.rows,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "embed_size": embed_size,
        },
        "config": {
            "softcap": args.softcap,
            "temperature": args.temperature,
            "allow_tf32": args.allow_tf32,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "block_k": args.block_k,
            "block_e": args.block_e,
            "num_warps": args.num_warps,
            "row_chunks": args.row_chunks,
            "max_acceptable_ratio": args.max_acceptable_ratio,
            "min_peak_reduction": args.min_peak_reduction,
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(),
        },
        "results": [
            asdict(r) for r in (full_res, lse_res, oracle_res, pass2_res)
            if r is not None
        ],
        "row_chunk_results": row_chunk_results,
        "best_row_chunk_fastest": best_row_chunk,
        "selected_row_chunk_memory_aware": selected_row_chunk,
        "ratios": {
            "pass2_vs_materialized_full_ms": ratio_vs_full,
            "pass2_vs_materialized_from_lse_ms": ratio_vs_lse,
            "pass2_vs_materialized_full_peak": (
                pass2_res.peak_mib / full_res.peak_mib
                if pass2_res is not None else None
            ),
            "best_row_chunk_vs_materialized_from_lse_ms": best_row_ratio,
            "selected_row_chunk_vs_materialized_from_lse_ms": selected_row_ratio,
            "best_row_chunk_vs_materialized_full_peak": (
                best_row_chunk["ratio_vs_materialized_full_peak"]
                if best_row_chunk is not None else None
            ),
            "selected_row_chunk_vs_materialized_from_lse_peak": (
                selected_row_chunk["ratio_vs_materialized_from_lse_peak"]
                if selected_row_chunk is not None else None
            ),
        },
        "correctness": {
            "max_abs_vs_materialized_full": max_abs_vs_full,
            "max_abs_vs_materialized_from_lse": max_abs_vs_lse,
            "max_abs_vs_fp32_oracle": (
                float((pass2 - oracle).abs().max().item())
                if pass2 is not None and oracle is not None else None
            ),
            "materialized_from_lse_vs_full": float(
                (lse_ref - full).abs().max().item()
            ),
            "materialized_from_lse_vs_fp32_oracle": (
                float((lse_ref - oracle).abs().max().item())
                if oracle is not None else None
            ),
        },
        "decision": decision,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
