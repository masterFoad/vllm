import torch
import math

def generate_hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Generate a Hadamard matrix of size n x n (n must be a power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]], device=device)
    
    h_half = generate_hadamard_matrix(n // 2, device)
    top = torch.cat([h_half, h_half], dim=1)
    bottom = torch.cat([h_half, -h_half], dim=1)
    return torch.cat([top, bottom], dim=0) / math.sqrt(2.0)

_hadamard_cache = {}

def get_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device, dtype)
    if key not in _hadamard_cache:
        _hadamard_cache[key] = generate_hadamard_matrix(n, device).to(dtype)
    return _hadamard_cache[key]
