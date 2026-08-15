import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2dCustom(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        return (x - mean) / std * self.weight + self.bias

class LayerNorm2dOpt(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def test():
    x = torch.randn(16, 32, 128, 128) # standard batch
    
    ln_custom = LayerNorm2dCustom(32)
    ln_opt = LayerNorm2dOpt(32)
    
    # Verify outputs are identical
    out_custom = ln_custom(x)
    out_opt = ln_opt(x)
    diff = torch.max(torch.abs(out_custom - out_opt)).item()
    print(f"Max absolute difference between custom and optimized: {diff:.2e}")
    
    # Warmup
    for _ in range(10):
        _ = ln_custom(x)
        _ = ln_opt(x)
        
    # Time Custom
    t0 = time.perf_counter()
    for _ in range(100):
        _ = ln_custom(x)
    t_custom = (time.perf_counter() - t0) * 1000.0
    
    # Time Optimized
    t0 = time.perf_counter()
    for _ in range(100):
        _ = ln_opt(x)
    t_opt = (time.perf_counter() - t0) * 1000.0
    
    print(f"Time for 100 passes:")
    print(f"  - Custom:    {t_custom:.2f} ms")
    print(f"  - Optimized: {t_opt:.2f} ms")
    print(f"  - Speedup:   {t_custom / t_opt:.2x}")

if __name__ == "__main__":
    test()
