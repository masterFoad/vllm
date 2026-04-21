import time
import torch
import triton
import triton.language as tl

# Mock the old and new kernels to measure the difference in execution time
# The old kernel runs for all positions (max_spec_len)
# The new kernel short-circuits on the first rejection

@triton.jit
def old_kernel(
    output_ptr,
    vocab_size,
    max_spec_len,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    
    # Old kernel: runs for all positions regardless of rejection
    for pos in range(max_spec_len):
        max_val = float("-inf")
        recovered_id = 0
        for v in range(0, vocab_size, BLOCK_SIZE):
            vocab_offset = v + tl.arange(0, BLOCK_SIZE)
            vocab_mask = vocab_offset < vocab_size
            
            # Simulate some math
            prob = tl.load(output_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_mask, other=0.0)
            score = prob * 0.5
            local_max, local_id = tl.max(score, axis=0, return_indices=True)
            
            if local_max > max_val:
                max_val = local_max
                recovered_id = v + local_id
                
        tl.store(output_ptr + req_idx, recovered_id)

@triton.jit
def new_kernel(
    output_ptr,
    vocab_size,
    max_spec_len,
    rejection_pos, # Simulate where the rejection happens
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    
    rejected = False
    for pos in range(max_spec_len):
        if not rejected:
            if pos == rejection_pos:
                # Reject! Do the heavy math ONCE for the rejected position
                rejected = True
                
                max_val = float("-inf")
                recovered_id = 0
                for v in range(0, vocab_size, BLOCK_SIZE):
                    vocab_offset = v + tl.arange(0, BLOCK_SIZE)
                    vocab_mask = vocab_offset < vocab_size
                    
                    # Simulate some math
                    prob = tl.load(output_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_mask, other=0.0)
                    score = prob * 0.5
                    local_max, local_id = tl.max(score, axis=0, return_indices=True)
                    
                    if local_max > max_val:
                        max_val = local_max
                        recovered_id = v + local_id
                        
                tl.store(output_ptr + req_idx, recovered_id)
            else:
                # Accept! Fast path, no heavy math
                tl.store(output_ptr + req_idx, 1)

def run_bench():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
        
    device = torch.device('cuda')
    batch_size = 128
    vocab_size = 128000
    max_spec_len = 8
    
    output = torch.zeros(batch_size * vocab_size, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(10):
        old_kernel[(batch_size,)](output, vocab_size, max_spec_len, BLOCK_SIZE=8192)
        new_kernel[(batch_size,)](output, vocab_size, max_spec_len, 0, BLOCK_SIZE=8192)
        
    torch.cuda.synchronize()
    
    # Benchmark Old (Eager computation for all 8 positions)
    start = time.perf_counter()
    for _ in range(1000):
        old_kernel[(batch_size,)](output, vocab_size, max_spec_len, BLOCK_SIZE=8192)
    torch.cuda.synchronize()
    end = time.perf_counter()
    old_time = (end - start) * 1000 / 1000 # ms per iter
    
    # Benchmark New (Rejects at position 0 - worst case for old, best case for new)
    start = time.perf_counter()
    for _ in range(1000):
        new_kernel[(batch_size,)](output, vocab_size, max_spec_len, 0, BLOCK_SIZE=8192)
    torch.cuda.synchronize()
    end = time.perf_counter()
    new_time_pos0 = (end - start) * 1000 / 1000 # ms per iter
    
    # Benchmark New (Rejects at position 4 - average case)
    start = time.perf_counter()
    for _ in range(1000):
        new_kernel[(batch_size,)](output, vocab_size, max_spec_len, 4, BLOCK_SIZE=8192)
    torch.cuda.synchronize()
    end = time.perf_counter()
    new_time_pos4 = (end - start) * 1000 / 1000 # ms per iter
    
    print(f"Old Eager Path (8 positions): {old_time:.4f} ms")
    print(f"New Lazy Path (Reject at pos 0): {new_time_pos0:.4f} ms ({old_time/new_time_pos0:.2f}x speedup)")
    print(f"New Lazy Path (Reject at pos 4): {new_time_pos4:.4f} ms ({old_time/new_time_pos4:.2f}x speedup)")

if __name__ == '__main__':
    run_bench()