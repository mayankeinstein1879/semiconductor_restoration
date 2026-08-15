import os
import sys
import glob
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import get_model

def run_reproducibility_test():
    print("=== STARTING REPRODUCIBILITY & DETERMINISM TEST ===")
    
    # 1. Config and Paths
    config_path = "configs/advanced_restoration_v1.yaml"
    checkpoint_path = "experiments/advanced_restoration_v1/checkpoints/best.pth"
    test_dir = "C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public/Test_NoisyLR/NoisyLR"
    out_dir = "reports/reproducibility_test_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    
    # 2. Instantiate and load model
    print("Loading optimized model from checkpoint...")
    model = get_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict directly (the optimized LayerNorm maintains 100% key and shape compatibility)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")
    
    # 3. Load subset of test images (first 5 images)
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))[:5]
    print(f"Testing on {len(test_files)} images...")
    
    norm_stats = ckpt.get("normalization", {})
    norm_mean = norm_stats.get("mean", config["dataset"].get("global_mean", 0.432994))
    norm_std = norm_stats.get("std", config["dataset"].get("global_std", 0.202202))
    
    for f in test_files:
        basename = os.path.basename(f)
        
        # Load and standardize
        img = np.load(f).astype(np.float32)
        std_img = (img - norm_mean) / norm_std
        input_tensor = torch.from_numpy(std_img).unsqueeze(0).unsqueeze(0).to(device)
        
        # Run inference twice to check determinism
        with torch.no_grad():
            out1 = model(input_tensor).squeeze().cpu().numpy()
            out2 = model(input_tensor).squeeze().cpu().numpy()
            
        # Checks:
        # A. Determinism
        diff = np.max(np.abs(out1 - out2))
        assert diff == 0.0, f"Determinism check failed for {basename}! Diff: {diff}"
        
        # B. Shape and grayscale
        assert out1.ndim == 2, f"Expected 2D grayscale output, got {out1.ndim}D for {basename}"
        assert out1.shape == (256, 256), f"Expected shape (256, 256), got {out1.shape} for {basename}"
        
        # C. Value range [0,1] and NaN/Inf
        assert not np.isnan(out1).any() and not np.isinf(out1).any(), f"NaN or Inf found in output of {basename}!"
        assert out1.min() >= 0.0 and out1.max() <= 1.0, f"Range violation for {basename}: [{out1.min()}, {out1.max()}]"
        
        # Save output
        np.save(os.path.join(out_dir, basename), out1)
        print(f"  - Passed: {basename} (Identical runs, Shape: {out1.shape}, Range: [{out1.min():.4f}, {out1.max():.4f}])")
        
    print("=== ALL REPRODUCIBILITY CHECKS PASSED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    run_reproducibility_test()
