import time
import random

def finalize(length_blocks, *, match_found, use_eagle, block_size, alignment_tokens):
    if not match_found and block_size != alignment_tokens:
        num_tokens = length_blocks * block_size
        length_blocks = (num_tokens // alignment_tokens) * alignment_tokens // block_size

    if use_eagle and length_blocks > 0:
        length_blocks -= 1
        if block_size != alignment_tokens:
            num_tokens = length_blocks * block_size
            length_blocks = (num_tokens // alignment_tokens) * alignment_tokens // block_size

    return length_blocks

def old_vllm_find_longest_cache_hit(
    max_num_blocks,
    block_hashes,
    is_cached_func,
    sliding_window_contiguous_blocks,
    block_size,
    alignment_tokens,
    use_eagle
):
    W = sliding_window_contiguous_blocks
    num_contiguous_blocks = 0
    match_found = False
    length_blocks = 0

    for i in range(max_num_blocks - 1, -1, -1):
        if is_cached_func(block_hashes[i]):
            if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                post_pop_blocks = i if use_eagle else i + 1
                if (post_pop_blocks * block_size) % alignment_tokens != 0:
                    continue

            num_contiguous_blocks += 1

            if num_contiguous_blocks >= W:
                length_blocks = i + num_contiguous_blocks
                match_found = True
                break
        else:
            num_contiguous_blocks = 0

    if not match_found:
        length_blocks = num_contiguous_blocks

    return finalize(
        length_blocks,
        match_found=match_found,
        use_eagle=use_eagle,
        block_size=block_size,
        alignment_tokens=alignment_tokens,
    )

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
        return finalize(
            end_index + 1,
            match_found=True,
            use_eagle=use_eagle,
            block_size=block_size,
            alignment_tokens=alignment_tokens
        )
        
    num_contiguous_blocks = 0
    for j in range(max_num_blocks):
        if is_cached_func(block_hashes[j]):
            num_contiguous_blocks += 1
        else:
            break
            
    return finalize(
        num_contiguous_blocks,
        match_found=False,
        use_eagle=use_eagle,
        block_size=block_size,
        alignment_tokens=alignment_tokens
    )

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
    
    # Benchmark Old vLLM O(M) Scan
    start = time.perf_counter()
    for _ in range(iterations):
        old_vllm_find_longest_cache_hit(
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
    print(f"Old vLLM O(M) Scan Time: {old_time:.4f} ms per lookup")
    print(f"New (Optimized Skip) Time: {new_time:.4f} ms per lookup")
    print(f"Speedup: {old_time / new_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
