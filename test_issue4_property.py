import random
import pytest

def brute_force_find_longest_cache_hit(
    max_num_blocks,
    block_hashes,
    is_cached_func,
    sliding_window_contiguous_blocks,
    block_size,
    alignment_tokens,
    use_eagle
):
    # Phase 1: Find the rightmost valid window of length W
    W = sliding_window_contiguous_blocks
    
    for end_index in range(max_num_blocks - 1, W - 2, -1):
        post_pop_blocks = end_index if use_eagle else end_index + 1
        if block_size != alignment_tokens and (post_pop_blocks * block_size) % alignment_tokens != 0:
            continue
            
        # Check if all blocks in the window are cached
        all_cached = True
        for j in range(end_index - W + 1, end_index + 1):
            if not is_cached_func(block_hashes[j]):
                all_cached = False
                break
                
        if all_cached:
            return end_index + 1 # Return length
            
    # Phase 2: Prefix match from 0
    num_contiguous_blocks = 0
    for j in range(max_num_blocks):
        if is_cached_func(block_hashes[j]):
            num_contiguous_blocks += 1
        else:
            break
            
    # Apply alignment and eagle logic to prefix match
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
        
    # Phase 2: Prefix match from 0
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

@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("block_size, alignment_tokens", [(16, 16), (16, 32), (1, 2)])
@pytest.mark.parametrize("W", [1, 2, 5, 10])
@pytest.mark.parametrize("hit_rate", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_property_sliding_window_skip(use_eagle, block_size, alignment_tokens, W, hit_rate):
    random.seed(42)
    for _ in range(100): # 100 random arrays per config
        max_num_blocks = random.randint(W, 100)
        block_hashes = list(range(max_num_blocks))
        
        # Generate random hit/miss array
        cache_state = {h: random.random() < hit_rate for h in block_hashes}
        
        # Sometimes force a prefix match but no full window
        if random.random() < 0.2 and max_num_blocks > W:
            for i in range(W - 1):
                cache_state[block_hashes[i]] = True
            for i in range(W - 1, max_num_blocks):
                cache_state[block_hashes[i]] = False
                
        def is_cached_func(h):
            return cache_state[h]
            
        brute_force_len = brute_force_find_longest_cache_hit(
            max_num_blocks, block_hashes, is_cached_func, W, block_size, alignment_tokens, use_eagle
        )
        
        optimized_len = optimized_find_longest_cache_hit(
            max_num_blocks, block_hashes, is_cached_func, W, block_size, alignment_tokens, use_eagle
        )
        
        assert brute_force_len == optimized_len, f"Mismatch! BF: {brute_force_len}, Opt: {optimized_len}. Config: W={W}, eagle={use_eagle}, bs={block_size}, align={alignment_tokens}, cache={cache_state}"

if __name__ == "__main__":
    pytest.main(["-v", "test_issue4_property.py"])
