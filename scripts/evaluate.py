import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import yaml
import numpy as np
import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import SemiconductorDataset
from models.factory import get_model
from metrics.metrics import compute_all_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline_run",
        help="Experiment name for saving reports"
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Instantiate model
    model = get_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Setup dataset and loader
    val_dataset = SemiconductorDataset(
        dataset_root=config["dataset"]["root"],
        split_file=config["dataset"]["val_split"],
        mode="val",
        normalize=config["dataset"]["normalize"],
        global_mean=config["dataset"]["global_mean"],
        global_std=config["dataset"]["global_std"],
        augment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # process one-by-one for precise timing and visualization
        shuffle=False
    )
    
    # Outputs folders
    save_dir = os.path.join("experiments", args.experiment_name, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    results = []
    times = []
    
    # We want to visualize the same samples as the bicubic baseline where possible.
    # The filenames are read from val_split.txt
    val_filenames = val_dataset.filenames
    viz_indices = np.linspace(0, len(val_filenames)-1, args.num_visualizations, dtype=int)
    
    # Pre-warm GPU if available
    if device.type == "cuda":
        dummy = torch.randn(1, 1, 128, 128).to(device)
        for _ in range(10):
            _ = model(dummy)
        torch.cuda.synchronize()
        
    print("Running validation inference...")
    for idx, (lr_tensor, gt_tensor) in enumerate(val_loader):
        fname = val_filenames[idx]
        lr_tensor = lr_tensor.to(device)
        
        # Timing validation inference
        t_start = time.perf_counter()
        with torch.no_grad():
            pred_tensor = model(lr_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_end = time.perf_counter()
        times.append((t_end - t_start) * 1000.0) # ms
        
        # Convert to NumPy
        pred_np = pred_tensor.squeeze(0).squeeze(0).cpu().numpy()
        gt_np = gt_tensor.squeeze(0).squeeze(0).cpu().numpy()
        
        # Load raw degraded for visualization/metrics
        raw_lr_path = os.path.join(config["dataset"]["root"], "train", "train", "NoisyLR", fname)
        raw_lr = np.load(raw_lr_path)
        
        # Compute metrics
        m = compute_all_metrics(gt_np, pred_np)
        m["filename"] = fname
        results.append(m)
        
        # Save visualization for selected samples
        if idx in viz_indices:
            basename = os.path.splitext(fname)[0]
            abs_diff = np.abs(pred_np - gt_np)
            
            # Bicubic upsampling for baseline comparison (Approach B: clip -> bicubic -> clip)
            raw_lr_clipped = np.clip(raw_lr, 0.0, 1.0)
            bicubic_baseline = cv2.resize(raw_lr_clipped, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_CUBIC)
            bicubic_baseline = np.clip(bicubic_baseline, 0.0, 1.0)
            
            # Setup 5-column visualization
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Col 1: Degraded Input
            im0 = axes[0].imshow(raw_lr, cmap='gray')
            axes[0].set_title(f"Degraded 128x128\nRange: [{raw_lr.min():.2f}, {raw_lr.max():.2f}]")
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Col 2: Bicubic Baseline
            im1 = axes[1].imshow(bicubic_baseline, cmap='gray', vmin=0, vmax=1)
            # Compute baseline metrics for title
            m_base = compute_all_metrics(gt_np, bicubic_baseline)
            axes[1].set_title(f"Bicubic Baseline\nPSNR: {m_base['PSNR']:.2f} dB")
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Col 3: Neural Restoration
            im2 = axes[2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f"Residual U-Net\nPSNR: {m['PSNR']:.2f} dB")
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Col 4: Ground Truth
            im3 = axes[3].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f"Ground Truth 256x256")
            axes[3].axis('off')
            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
            
            # Col 5: Absolute Error
            im4 = axes[4].imshow(abs_diff, cmap='hot', vmin=0, vmax=0.5)
            axes[4].set_title(f"Abs Error (Mean: {abs_diff.mean():.4f})")
            axes[4].axis('off')
            fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"val_{basename}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Generate zoomed-in crop comparison of high-frequency structures
            # We select a 64x64 region in the center of the 256x256 image
            h, w = gt_np.shape
            ch, cw = h // 2, w // 2
            crop_r = 32
            
            gt_crop = gt_np[ch-crop_r:ch+crop_r, cw-crop_r:cw+crop_r]
            pred_crop = pred_np[ch-crop_r:ch+crop_r, cw-crop_r:cw+crop_r]
            bicubic_crop = bicubic_baseline[ch-crop_r:ch+crop_r, cw-crop_r:cw+crop_r]
            diff_crop = abs_diff[ch-crop_r:ch+crop_r, cw-crop_r:cw+crop_r]
            
            fig_crop, axes_crop = plt.subplots(1, 4, figsize=(16, 4))
            
            axes_crop[0].imshow(bicubic_crop, cmap='gray', vmin=0, vmax=1)
            axes_crop[0].set_title("Bicubic Crop (Zoom)")
            axes_crop[0].axis('off')
            
            axes_crop[1].imshow(pred_crop, cmap='gray', vmin=0, vmax=1)
            axes_crop[1].set_title("Neural Crop (Zoom)")
            axes_crop[1].axis('off')
            
            axes_crop[2].imshow(gt_crop, cmap='gray', vmin=0, vmax=1)
            axes_crop[2].set_title("GT Crop (Zoom)")
            axes_crop[2].axis('off')
            
            axes_crop[3].imshow(diff_crop, cmap='hot', vmin=0, vmax=0.3)
            axes_crop[3].set_title("Abs Error Crop")
            axes_crop[3].axis('off')
            
            plt.tight_layout()
            save_crop_path = os.path.join(save_dir, f"val_{basename}_crop.png")
            plt.savefig(save_crop_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Generated visualization and zoom crop for {fname}")
            
    # Compute stats
    df = pd.DataFrame(results)
    df_metrics = df[["PSNR", "SSIM", "MAE", "MSE"]]
    
    summary = {
        "Metric": ["PSNR", "SSIM", "MAE", "MSE"],
        "Mean": [df_metrics["PSNR"].mean(), df_metrics["SSIM"].mean(), df_metrics["MAE"].mean(), df_metrics["MSE"].mean()],
        "Median": [df_metrics["PSNR"].median(), df_metrics["SSIM"].median(), df_metrics["MAE"].median(), df_metrics["MSE"].median()],
        "Std": [df_metrics["PSNR"].std(), df_metrics["SSIM"].std(), df_metrics["MAE"].std(), df_metrics["MSE"].std()]
    }
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("reports/validation_metrics.csv", index=False)
    
    print("\nModel Evaluation Summary on Validation Split:")
    print(df_summary.to_string(index=False))
    
    # Timing
    total_time = sum(times)
    avg_time = np.mean(times)
    print(f"\nInference Speed:")
    print(f"  - Total validation time: {total_time:.2f} ms")
    print(f"  - Average time per image: {avg_time:.4f} ms")
    print(f"  - Images per second: {1000.0 / avg_time:.2f}")

if __name__ == "__main__":
    main()
