import torch
import pytest
from vllm.model_executor.layers.hadamard import get_hadamard_matrix

def simulate_quantization(x, bits=8):
    max_val = x.abs().max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-8)
    q_max = (2 ** (bits - 1)) - 1
    scale = max_val / q_max
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -q_max, q_max)
    return x_q * scale

def test_mla_orthogonal_stabilization():
    torch.manual_seed(42)
    batch = 16
    seq = 128
    d = 512
    
    # Simulate queries and KV latents
    q = torch.randn(batch, seq, d)
    c_kv = torch.randn(batch, seq, d)
    
    # Inject massive outliers into c_kv
    outlier_dims = torch.randperm(d)[:5]
    c_kv[:, :, outlier_dims] *= 100.0
    
    # 1. Standard MLA Attention (with simulated FP8 quantization)
    c_kv_quant_std = simulate_quantization(c_kv, bits=8)
    attn_std = torch.matmul(q, c_kv_quant_std.transpose(-1, -2))
    
    # 2. Stabilized MLA Attention
    H = get_hadamard_matrix(d, q.device, q.dtype)
    
    # Transform KV latents
    c_kv_rotated = torch.matmul(c_kv, H.t())
    c_kv_quant_stab = simulate_quantization(c_kv_rotated, bits=8)
    
    # Transform queries
    q_rotated = torch.matmul(q, H)
    
    # Compute attention
    attn_stab = torch.matmul(q_rotated, c_kv_quant_stab.transpose(-1, -2))
    
    # 3. Exact Attention (No quantization)
    attn_exact = torch.matmul(q, c_kv.transpose(-1, -2))
    
    # Compare MSE
    mse_std = torch.nn.functional.mse_loss(attn_exact, attn_std).item()
    mse_stab = torch.nn.functional.mse_loss(attn_exact, attn_stab).item()
    
    print(f"Standard MSE: {mse_std:.4f}")
    print(f"Stabilized MSE: {mse_stab:.4f}")
    print(f"Improvement: {mse_std / mse_stab:.2f}x")
    
    assert mse_stab < mse_std / 10, "Stabilized MSE should be at least 10x lower"

if __name__ == "__main__":
    test_mla_orthogonal_stabilization()
