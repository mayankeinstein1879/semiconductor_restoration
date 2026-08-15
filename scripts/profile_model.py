import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from models.advanced_restoration import AdvancedRestorationv1

def main():
    device = torch.device("cpu")
    model = AdvancedRestorationv1(base_channels=16).to(device)
    model.eval()
    
    # 128x128 single-image inference input
    x = torch.randn(1, 1, 128, 128, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        
    print("=== PROFILING ADVANCED MODEL FOR SINGLE IMAGE INFERENCE ===")
    
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        for _ in range(50):
            _ = model(x)
            
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

if __name__ == "__main__":
    main()
