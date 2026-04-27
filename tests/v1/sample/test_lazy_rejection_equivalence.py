import torch
import pytest
import triton
import triton.language as tl
from benchmark_real_lazy_rejection import sample_recovered_tokens_kernel, old_rejection_random_sample_kernel, new_rejection_random_sample_kernel

def test_seeded_equivalence():
    torch.manual_seed(42)
    device = torch.device('cuda')
    batch_size = 16
    vocab_size = 32000
    max_spec_len = 8
    num_tokens = batch_size * max_spec_len
    
    cu_num_draft_tokens = torch.arange(max_spec_len, num_tokens + 1, max_spec_len, dtype=torch.int32, device=device)
    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32, device=device)
    target_probs = torch.rand((num_tokens, vocab_size), dtype=torch.float32, device=device)
    
    # Generate inv_q exactly as vLLM does
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    inv_q = q.reciprocal()
    
    # Random uniform probs to trigger a mix of accepts and rejects
    uniform_probs = torch.rand(num_tokens, dtype=torch.float32, device=device)
    
    # Old Path
    output_token_ids_old = torch.zeros((batch_size, max_spec_len + 1), dtype=torch.int32, device=device)
    recovered_token_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids, cu_num_draft_tokens, draft_token_ids, target_probs, inv_q, vocab_size, BLOCK_SIZE=8192
    )
    old_rejection_random_sample_kernel[(batch_size,)](
        output_token_ids_old, cu_num_draft_tokens, draft_token_ids, target_probs, recovered_token_ids, uniform_probs, max_spec_len, vocab_size
    )
    
    # New Path
    output_token_ids_new = torch.zeros((batch_size, max_spec_len + 1), dtype=torch.int32, device=device)
    
    new_rejection_random_sample_kernel[(batch_size,)](
        output_token_ids_new, cu_num_draft_tokens, draft_token_ids, target_probs, inv_q, uniform_probs, max_spec_len, vocab_size, BLOCK_SIZE=8192
    )
    
    # Assert exact equivalence
    assert torch.equal(output_token_ids_old, output_token_ids_new), "Lazy rejection sampling output does not match eager output!"
    print("Seeded equivalence test passed!")

if __name__ == "__main__":
    test_seeded_equivalence()
