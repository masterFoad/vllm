import torch
import math

def simulate_quantization(x, bits=8):
    """Simulate symmetric per-token quantization to `bits` bits."""
    # x shape: [batch, seq, d]
    # Find max absolute value per token
    max_val = x.abs().max(dim=-1, keepdim=True)[0]
    
    # Avoid division by zero
    max_val = torch.clamp(max_val, min=1e-8)
    
    # Calculate scale
    q_max = (2 ** (bits - 1)) - 1
    scale = max_val / q_max
    
    # Quantize
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, -q_max, q_max)
    
    # Dequantize
    x_dq = x_q * scale
    return x_dq

def generate_hadamard_matrix(n):
    """Generate a Hadamard matrix of size n x n (n must be a power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]])
    
    h_half = generate_hadamard_matrix(n // 2)
    top = torch.cat([h_half, h_half], dim=1)
    bottom = torch.cat([h_half, -h_half], dim=1)
    return torch.cat([top, bottom], dim=0) / math.sqrt(2.0)

def run_prototype():
    torch.manual_seed(42)
    
    batch = 128
    seq = 1024
    d = 512
    
    print(f"Generating synthetic latent vectors [batch={batch}, seq={seq}, d={d}]...")
    # Normal distribution
    x = torch.randn(batch, seq, d)
    
    # Inject massive outliers (e.g., 100x magnitude) into a few random dimensions
    num_outliers = 5
    outlier_dims = torch.randperm(d)[:num_outliers]
    x[:, :, outlier_dims] *= 100.0
    
    print(f"Injected outliers at dimensions: {outlier_dims.tolist()}")
    
    # 1. Standard Quantization
    print("\n--- Standard 8-bit Quantization ---")
    x_dq_standard = simulate_quantization(x, bits=8)
    mse_standard = torch.nn.functional.mse_loss(x, x_dq_standard).item()
    print(f"MSE: {mse_standard:.6f}")
    
    # 2. Orthogonal Stabilization (Hadamard)
    print("\n--- Orthogonal Stabilization (Hadamard) + 8-bit Quantization ---")
    # Generate Hadamard matrix
    H = generate_hadamard_matrix(d).to(x.device)
    
    # Transform: x' = x * H^T
    x_rotated = torch.matmul(x, H.t())
    
    # Quantize
    x_rotated_dq = simulate_quantization(x_rotated, bits=8)
    
    # Inverse Transform: x_reconstructed = x'_dq * H
    # Since H is orthogonal, H^-1 = H^T. So we multiply by H.
    x_dq_stabilized = torch.matmul(x_rotated_dq, H)
    
    mse_stabilized = torch.nn.functional.mse_loss(x, x_dq_stabilized).item()
    print(f"MSE: {mse_stabilized:.6f}")
    
    print(f"\nImprovement: {mse_standard / mse_stabilized:.2f}x lower MSE")

if __name__ == "__main__":
    run_prototype()