import time
import random

def brute_force_find_longest_cache_hit(
    max_num_blocks,
    block_hashes,
    is_cached_func,
    sliding_window_contiguous_blocks,
    block_size,
    alignment_tokens,
    use_eagle
):
    W = sliding_window_contiguous_blocks
    for end_index in range(max_num_blocks - 1, W - 2, -1):
        post_pop_blocks = end_index if use_eagle else end_index + 1
        if block_size != alignment_tokens and (post_pop_blocks * block_size) % alignment_tokens != 0:
            continue
        all_cached = True
        for j in range(end_index - W + 1, end_index + 1):
            if not is_cached_func(block_hashes[j]):
                all_cached = False
                break
        if all_cached:
            return end_index + 1
    num_contiguous_blocks = 0
    for j in range(max_num_blocks):
        if is_cached_func(block_hashes[j]):
            num_contiguous_blocks += 1
        else:
            break
    if use_eagle and num_contiguous_blocks > 0:
        num_contiguous_blocks -= 1
    if block_size != alignment_tokens:
        num_tokens = num_contiguous_blocks * block_size
        num_tokens = (num_tokens // alignment_tokens) * alignment_tokens
        num_contiguous_blocks = num_tokens // block_size
    return num_contiguous_blocks

def optimized_find_longest_cache_hit(
    max_num_blocks,
    block_hashes,
    is_cached_func,
    sliding_window_contiguous_blocks,
    block_size,
    alignment_tokens,
    use_eagle
):
    W = sliding_window_contiguous_blocks
    end_index = max_num_blocks - 1
    match_found = False
    while end_index >= W - 1:
        post_pop_blocks = end_index if use_eagle else end_index + 1
        if block_size != alignment_tokens and (post_pop_blocks * block_size) % alignment_tokens != 0:
            end_index -= 1
            continue
        miss_index = -1
        for j in range(end_index - W + 1, end_index + 1):
            if not is_cached_func(block_hashes[j]):
                miss_index = j
                break
        if miss_index == -1:
            match_found = True
            break
        else:
            end_index = miss_index - 1
    if match_found:
        return end_index + 1
    num_contiguous_blocks = 0
    for j in range(max_num_blocks):
        if is_cached_func(block_hashes[j]):
            num_contiguous_blocks += 1
        else:
            break
    if use_eagle and num_contiguous_blocks > 0:
        num_contiguous_blocks -= 1
    if block_size != alignment_tokens:
        num_tokens = num_contiguous_blocks * block_size
        num_tokens = (num_tokens // alignment_tokens) * alignment_tokens
        num_contiguous_blocks = num_tokens // block_size
    return num_contiguous_blocks

def run_benchmark():
    random.seed(42)
    max_num_blocks = 8192 # Long prompt (e.g., 128k context with block size 16)
    block_hashes = list(range(max_num_blocks))
    W = 128 # Large sliding window
    block_size = 16
    alignment_tokens = 16
    use_eagle = False
    
    # Low hit rate scenario (e.g., 10% hit rate)
    hit_rate = 0.1
    cache_state = {h: random.random() < hit_rate for h in block_hashes}
    
    # Ensure no full window match to force worst-case scanning
    for i in range(max_num_blocks - W + 1):
        cache_state[block_hashes[i + W - 1]] = False
        
    def is_cached_func(h):
        return cache_state[h]
        
    iterations = 1000
    
    # Benchmark Brute Force (Old)
    start = time.perf_counter()
    for _ in range(iterations):
        brute_force_find_longest_cache_hit(
            max_num_blocks, block_hashes, is_cached_func, W, block_size, alignment_tokens, use_eagle
        )
    end = time.perf_counter()
    old_time = (end - start) * 1000 / iterations
    
    # Benchmark Optimized (New)
    start = time.perf_counter()
    for _ in range(iterations):
        optimized_find_longest_cache_hit(
            max_num_blocks, block_hashes, is_cached_func, W, block_size, alignment_tokens, use_eagle
        )
    end = time.perf_counter()
    new_time = (end - start) * 1000 / iterations
    
    print(f"Benchmark: Long Prompt ({max_num_blocks} blocks), Low Hit Rate ({hit_rate*100}%), Window Size ({W})")
    print(f"Old (Brute Force) Time: {old_time:.4f} ms per lookup")
    print(f"New (Optimized Skip) Time: {new_time:.4f} ms per lookup")
    print(f"Speedup: {old_time / new_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
