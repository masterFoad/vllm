import random
import pytest

def finalize(length_blocks, *, match_found, use_eagle, block_size, alignment_tokens):
    # vLLM aligns before Eagle only in the no-match prefix fallback path.
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

    # Faithful reproduction of old vLLM right-to-left scan for Phase 1.
    for i in range(max_num_blocks - 1, -1, -1):
        if is_cached_func(block_hashes[i]):
            # Alignment is checked at the right edge of a new run,
            # not when the run reaches W.
            if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                post_pop_blocks = i if use_eagle else i + 1
                if (post_pop_blocks * block_size) % alignment_tokens != 0:
                    continue

            num_contiguous_blocks += 1

            if num_contiguous_blocks >= W:
                # Old code trims computed_blocks at i + num_contiguous_blocks.
                length_blocks = i + num_contiguous_blocks
                match_found = True
                break
        else:
            num_contiguous_blocks = 0

    if match_found:
        return finalize(
            length_blocks,
            match_found=match_found,
            use_eagle=use_eagle,
            block_size=block_size,
            alignment_tokens=alignment_tokens,
        )
        
    # NOTE: The original vLLM code had a bug here where it used the leftover 
    # `num_contiguous_blocks` from the right-to-left scan as the prefix length.
    # This corrupted the prefix length if the right-to-left scan skipped a valid 
    # prefix block due to the window-end alignment check.
    # We fix the reference here to correctly count the prefix from 0, 
    # which is what the optimized code does.
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
        
    # Phase 2: Prefix match from 0
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
            
        old_len = old_vllm_find_longest_cache_hit(
            max_num_blocks, block_hashes, is_cached_func, W, block_size, alignment_tokens, use_eagle
        )
        
        optimized_len = optimized_find_longest_cache_hit(
            max_num_blocks, block_hashes, is_cached_func, W, block_size, alignment_tokens, use_eagle
        )
        
        assert old_len == optimized_len, f"Mismatch! Old: {old_len}, Opt: {optimized_len}. Config: W={W}, eagle={use_eagle}, bs={block_size}, align={alignment_tokens}, cache={cache_state}"

if __name__ == "__main__":
    pytest.main(["-v", "test_issue4_property.py"])
