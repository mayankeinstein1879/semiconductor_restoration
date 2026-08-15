import os
import sys
import argparse
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import get_model

def main():
    parser = argparse.ArgumentParser(description="Standalone inference script for semiconductor restoration.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input degraded .npy images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the restored 2x images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    args = parser.parse_args()
    
    # 1. Verification of directories
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Discover files
    input_files = sorted([
        f for f in glob.glob(os.path.join(args.input_dir, "*.npy"))
    ])
    
    if len(input_files) == 0:
        print(f"Error: No .npy files found in input directory: {args.input_dir}")
        sys.exit(1)
        
    print(f"Discovered {len(input_files)} compatible semiconductor files for inference.")
    
    # 3. Load checkpoint and dynamically extract config & normalization statistics
    print(f"Loading checkpoint from: {args.checkpoint}...")
    device = torch.device("cpu")
    
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        print(f"Error: Could not load checkpoint file: {str(e)}")
        sys.exit(1)
        
    # Extract config
    config = ckpt.get("config")
    if not config:
        print("Warning: No config dict found inside checkpoint. Fallback to default Residual U-Net config.")
        config = {
            "model": {"name": "residual_unet", "base_channels": 16, "activation_type": "sigmoid"},
            "dataset": {"global_mean": 0.432994, "global_std": 0.202202}
        }
        
    # Extract normalization statistics
    norm_stats = ckpt.get("normalization", {})
    norm_mean = norm_stats.get("mean", config.get("dataset", {}).get("global_mean", 0.432994))
    norm_std = norm_stats.get("std", config.get("dataset", {}).get("global_std", 0.202202))
    
    print(f"Checkpoint Normalization Parameters Loaded:")
    print(f"  - Mean: {norm_mean:.6f}")
    print(f"  - Std:  {norm_std:.6f}")
    
    # 4. Instantiate Model
    model_name = config.get("model", {}).get("name", "residual_unet")
    print(f"Instantiating model architecture: '{model_name}'...")
    model = get_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # 5. Run CPU Batch Inference
    print("\nRunning inference pipeline on CPU...")
    inference_times = []
    output_values = []
    input_shapes = []
    output_shapes = []
    
    # Warmup
    warmup_tensor = torch.zeros(1, 1, 128, 128)
    for _ in range(5):
        _ = model(warmup_tensor)
        
    for idx, f in enumerate(input_files):
        basename = os.path.basename(f)
        
        # Load raw degraded image
        raw_img = np.load(f)
        if raw_img.ndim != 2:
            print(f"Warning: Skipping file '{basename}' because it is not 2D. Shape: {raw_img.shape}")
            continue
            
        input_shapes.append(raw_img.shape)
        
        # Check for NaNs/Infs
        if np.isnan(raw_img).any() or np.isinf(raw_img).any():
            print(f"Warning: Skipping file '{basename}' due to NaN/Inf values.")
            continue
            
        # Preprocess: standardize using training statistics
        std_img = (raw_img - norm_mean) / norm_std
        input_tensor = torch.from_numpy(std_img).unsqueeze(0).unsqueeze(0).float()
        
        # Run inference
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_t = model(input_tensor)
        t1 = time.perf_counter()
        
        inference_times.append((t1 - t0) * 1000.0) # in ms
        pred = pred_t.squeeze().numpy() # Shape (H_out, W_out)
        
        output_shapes.append(pred.shape)
        output_values.append(pred.flatten())
        
        # Save output as npy to preserve precision
        save_path = os.path.join(args.output_dir, basename)
        np.save(save_path, pred)
        
    # 6. Report Inference Statistics
    if len(output_values) == 0:
        print("Error: No images were successfully processed.")
        sys.exit(1)
        
    output_values = np.concatenate(output_values)
    
    print("\n=== INFERENCE EXECUTION COMPLETED ===")
    print(f"Total images processed:  {len(input_shapes)}")
    print(f"Input shape range:       Min {min(input_shapes)} | Max {max(input_shapes)}")
    print(f"Output shape range:      Min {min(output_shapes)} | Max {max(output_shapes)}")
    print(f"Output value range:      Min {output_values.min():.6f} | Max {output_values.max():.6f}")
    print(f"Output Mean / Std:       {output_values.mean():.6f} / {output_values.std():.6f}")
    
    avg_speed = np.mean(inference_times)
    total_time = np.sum(inference_times)
    print(f"Average CPU latency:     {avg_speed:.2f} ms/image")
    print(f"Throughput (FPS):        {1000.0 / avg_speed:.2f} images/sec")
    print(f"Total CPU execution:     {total_time:.2f} ms")
    print("=====================================\n")

if __name__ == "__main__":
    main()
