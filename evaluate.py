import os
import sys
import argparse
import time
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Subset

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SemiconductorDataset
from models.factory import get_model
from metrics.metrics import compute_all_metrics
import torch.nn.functional as F

def get_sobel_gradients(img, device):
    if isinstance(img, np.ndarray):
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    else:
        img_t = img.to(device)
        
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    gx = F.conv2d(img_t, kernel_x, padding=1)
    gy = F.conv2d(img_t, kernel_y, padding=1)
    
    return gx.squeeze().cpu().numpy(), gy.squeeze().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Standalone evaluation script for semiconductor restoration.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint file (.pth).")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    
    # 1. Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Load model configuration from checkpoint if available, otherwise config argument
    ckpt_config = ckpt.get("config", config)
    
    # 2. Instantiate Model
    print("Instantiating model...")
    model = get_model(ckpt_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Extract normalization parameters from checkpoint or config
    norm_stats = ckpt.get("normalization", {})
    norm_mean = norm_stats.get("mean", ckpt_config.get("dataset", {}).get("global_mean", 0.432994))
    norm_std = norm_stats.get("std", ckpt_config.get("dataset", {}).get("global_std", 0.202202))
    
    print(f"Loaded normalizations - Mean: {norm_mean:.6f} | Std: {norm_std:.6f}")
    
    # 3. Setup Dataset
    dataset_root = ckpt_config["dataset"]["root"]
    input_dir = os.path.join(dataset_root, "train", "train", "NoisyLR")
    target_dir = os.path.join(dataset_root, "train", "train", "GT")
    
    raw_dataset = SemiconductorDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        normalize=ckpt_config["dataset"].get("normalize", "standardize"),
        global_mean=norm_mean,
        global_std=norm_std,
        augment=False,
        scale_factor=ckpt_config["model"].get("scale_factor", 2)
    )
    
    # Splits filtering
    val_split_path = ckpt_config["dataset"].get("val_split")
    if val_split_path and os.path.exists(val_split_path):
        with open(val_split_path, "r") as f:
            val_names = set(line.strip() for line in f if line.strip())
        val_indices = [i for i, f in enumerate(raw_dataset.filenames) if f in val_names]
        print(f"Loaded validation split: {len(val_indices)} samples.")
    else:
        # Default fallback to 10% validation split based on fixed random seed
        import random
        num_samples = len(raw_dataset)
        indices = list(range(num_samples))
        random.seed(42)
        random.shuffle(indices)
        split_idx = int(0.9 * num_samples)
        val_indices = indices[split_idx:]
        print(f"No validation split file found. Using default 10% partition: {len(val_indices)} samples.")
        
    val_loader = DataLoader(
        Subset(raw_dataset, val_indices),
        batch_size=1, # Single-image evaluation for latency benchmarking
        shuffle=False
    )
    
    # 4. Evaluation Loop
    print("\nRunning validation evaluation on CPU...")
    psnrs, ssims, maes, mses, grad_maes = [], [], [], [], []
    inference_times = []
    
    # Warmup
    warmup_tensor = torch.zeros(1, 1, 128, 128)
    for _ in range(10):
        _ = model(warmup_tensor)
        
    for lr_tensor, gt_tensor in val_loader:
        gt_np = gt_tensor.squeeze().numpy()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(lr_tensor).squeeze().numpy()
        t1 = time.perf_counter()
        
        inference_times.append((t1 - t0) * 1000.0) # in ms
        
        # Calculate standard reconstruction metrics
        m = compute_all_metrics(gt_np, pred)
        psnrs.append(m["PSNR"])
        ssims.append(m["SSIM"])
        maes.append(m["MAE"])
        mses.append(m["MSE"])
        
        # Calculate directional Sobel gradient error
        gx_gt, gy_gt = get_sobel_gradients(gt_np, device)
        gx_pred, gy_pred = get_sobel_gradients(pred, device)
        grad_mae = np.mean(np.abs(gx_pred - gx_gt) + np.abs(gy_pred - gy_gt))
        grad_maes.append(grad_mae)
        
    # 5. Output Results
    print("\n=== MODEL EVALUATION SUMMARY ON VALIDATION SPLIT ===")
    print(f"Metric      Mean        Median      Std")
    print(f"  PSNR      {np.mean(psnrs):.6f}   {np.median(psnrs):.6f}   {np.std(psnrs):.6f}")
    print(f"  SSIM      {np.mean(ssims):.6f}   {np.median(ssims):.6f}   {np.std(ssims):.6f}")
    print(f"   MAE      {np.mean(maes):.6f}   {np.median(maes):.6f}   {np.std(maes):.6f}")
    print(f"   MSE      {np.mean(mses):.6f}   {np.median(mses):.6f}   {np.std(mses):.6f}")
    print(f"  Grad MAE  {np.mean(grad_maes):.6f}   {np.median(grad_maes):.6f}   {np.std(grad_maes):.6f}")
    
    avg_speed = np.mean(inference_times)
    total_time = np.sum(inference_times)
    print(f"\nInference Speed:")
    print(f"  - Total validation time: {total_time:.2f} ms")
    print(f"  - Average time per image: {avg_speed:.4f} ms")
    print(f"  - Throughput: {1000.0 / avg_speed:.2f} images/sec (FPS)")
    print("===================================================\n")

if __name__ == "__main__":
    main()
