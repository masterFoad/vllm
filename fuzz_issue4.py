import sys
from unittest.mock import MagicMock

# Bypass CUDA imports
sys.modules["vllm._C"] = MagicMock()
sys.modules["vllm._C_stable_libtorch"] = MagicMock()
sys.modules["vllm._C.ops"] = MagicMock()
sys.modules["vllm._C.custom_ar"] = MagicMock()

import torch
import random
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.utils.math_utils import cdiv


def naive_reference_implementation(
    block_hashes, max_length, pool, spec, use_eagle, alignment_tokens
):
    """The known-correct sequential reverse scan implementation."""
    block_size = spec.block_size
    sliding_window_contiguous_blocks = cdiv(spec.sliding_window - 1, block_size)
    if use_eagle:
        sliding_window_contiguous_blocks += 1

    max_num_blocks = max_length // block_size
    W = sliding_window_contiguous_blocks

    match_found = False
    final_blocks = [pool.null_block] * max_num_blocks

    for end_idx in range(max_num_blocks - 1, W - 2, -1):
        if (
            block_size != alignment_tokens
            and (end_idx + 1) * block_size % alignment_tokens != 0
        ):
            continue

        is_hit = True
        current_window = []
        for j in range(end_idx - W + 1, end_idx + 1):
            cached = pool.get_cached_block(block_hashes[j], [0])
            if not cached:
                is_hit = False
                break
            current_window.append(cached[0])

        if is_hit:
            for j, blk in enumerate(current_window):
                final_blocks[end_idx - W + 1 + j] = blk
            final_blocks = final_blocks[: end_idx + 1]
            match_found = True
            break

    if not match_found:
        num_contiguous = 0
        for j in range(max_num_blocks):
            cached = pool.get_cached_block(block_hashes[j], [0])
            if cached:
                final_blocks[j] = cached[0]
                num_contiguous += 1
            else:
                break
        final_blocks = final_blocks[:num_contiguous]
        while (
            block_size != alignment_tokens
            and len(final_blocks) * block_size % alignment_tokens != 0
        ):
            final_blocks.pop()

    if use_eagle and final_blocks:
        final_blocks.pop()

    return final_blocks


def run_differential_fuzzer():
    print("Starting Differential Correctness Fuzzer (5,000 trials)...")

    for trial in range(5000):
        block_size = random.choice([2, 4, 8, 16])
        sliding_window = random.randint(block_size * 2, block_size * 16)
        use_eagle = random.choice([True, False])
        if use_eagle:
            alignment_tokens = block_size
        else:
            alignment_tokens = random.choice(
                [block_size, block_size * 2, block_size * 4]
            )

        max_blocks = random.randint(10, 100)

        spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=sliding_window,
        )
        pool = BlockPool(
            num_gpu_blocks=max_blocks + 20,
            enable_caching=True,
            hash_block_size=block_size,
        )
        manager = SlidingWindowManager(
            spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0
        )

        hashes = [BlockHash(str(i).encode()) for i in range(max_blocks)]
        hit_mask = [random.random() < 0.4 for _ in range(max_blocks)]
        for i, hit in enumerate(hit_mask):
            if hit:
                pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(hashes[i], 0), pool.blocks[i]
                )

        opt_result = manager.find_longest_cache_hit(
            block_hashes=hashes,
            max_length=max_blocks * block_size,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            use_eagle=use_eagle,
            alignment_tokens=alignment_tokens,
        )[0]

        ref_result = naive_reference_implementation(
            hashes, max_blocks * block_size, pool, spec, use_eagle, alignment_tokens
        )

        if [b.block_id for b in opt_result] != [b.block_id for b in ref_result]:
            print(f"\nDIFF FOUND in trial {trial}!")
            sys.exit(1)

        if trial % 500 == 0:
            print(f"Trial {trial} passed...")

    print("\nSUCCESS: All 5,000 differential trials passed perfectly.")


if __name__ == "__main__":
    run_differential_fuzzer()
