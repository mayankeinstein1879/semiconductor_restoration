import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from metrics.metrics import compute_all_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bicubic Baseline")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public",
        help="Path to dataset root"
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="data/val_split.txt",
        help="Path to validation split file"
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=10,
        help="Number of validation samples to visualize"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check split file
    if not os.path.exists(args.val_split):
        print(f"Error: Validation split file not found at {args.val_split}")
        return
        
    with open(args.val_split, "r") as f:
        val_filenames = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(val_filenames)} validation filenames.")
    
    gt_dir = os.path.join(args.dataset_root, "train", "train", "GT")
    lr_dir = os.path.join(args.dataset_root, "train", "train", "NoisyLR")
    
    # Results accumulation
    results_a = [] # Approach A: raw degraded -> bicubic -> clip output -> metrics
    results_b1 = [] # Approach B1: clip degraded -> bicubic -> metrics (no final clipping)
    results_b2 = [] # Approach B2: clip degraded -> bicubic -> clip output -> metrics
    
    times_a = []
    times_b = []
    
    # For visualizations, we will pick 10 evenly spaced samples
    viz_indices = np.linspace(0, len(val_filenames)-1, args.num_visualizations, dtype=int)
    os.makedirs("reports/baseline_visualizations", exist_ok=True)
    
    for idx, fname in enumerate(val_filenames):
        gt_path = os.path.join(gt_dir, fname)
        lr_path = os.path.join(lr_dir, fname)
        
        if not os.path.exists(gt_path) or not os.path.exists(lr_path):
            print(f"Warning: File pair not found: {fname}")
            continue
            
        gt_img = np.load(gt_path)
        lr_img = np.load(lr_path)
        h_gt, w_gt = gt_img.shape
        
        # --- Evaluate Approach A: Raw degraded -> Bicubic -> Clip output ---
        t_start = time.perf_counter()
        # Bicubic upsampling
        bicubic_a = cv2.resize(lr_img, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
        # Clip output
        bicubic_a_clipped = np.clip(bicubic_a, 0.0, 1.0)
        t_end = time.perf_counter()
        times_a.append((t_end - t_start) * 1000.0) # ms
        
        metrics_a = compute_all_metrics(gt_img, bicubic_a_clipped)
        results_a.append(metrics_a)
        
        # --- Evaluate Approach B: Clip degraded -> Bicubic ---
        t_start = time.perf_counter()
        # Clip degraded input
        lr_img_clipped = np.clip(lr_img, 0.0, 1.0)
        # Bicubic upsampling
        bicubic_b = cv2.resize(lr_img_clipped, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
        t_end = time.perf_counter()
        times_b.append((t_end - t_start) * 1000.0) # ms
        
        # B1: without final clipping (for metrics compatibility, we clip metrics calculation or let it evaluate)
        # We enforce clip for metric computation to avoid PSNR NaN, but we check values
        metrics_b1 = compute_all_metrics(gt_img, np.clip(bicubic_b, 0.0, 1.0)) # we clip output for SSIM/PSNR calculation safety
        results_b1.append(metrics_b1)
        
        # B2: clip output
        bicubic_b_clipped = np.clip(bicubic_b, 0.0, 1.0)
        metrics_b2 = compute_all_metrics(gt_img, bicubic_b_clipped)
        results_b2.append(metrics_b2)
        
        # --- Save Visualizations for Selected Samples ---
        if idx in viz_indices:
            basename = os.path.splitext(fname)[0]
            abs_diff = np.abs(bicubic_a_clipped - gt_img)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            im0 = axes[0].imshow(lr_img, cmap='gray')
            axes[0].set_title(f"Degraded 128x128\nRange: [{lr_img.min():.2f}, {lr_img.max():.2f}]")
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(bicubic_a_clipped, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f"Bicubic (Clipped Output)\nPSNR: {metrics_a['PSNR']:.2f} dB")
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f"Ground Truth 256x256")
            axes[2].axis('off')
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            im3 = axes[3].imshow(abs_diff, cmap='hot', vmin=0, vmax=0.5)
            axes[3].set_title(f"Abs Error (Mean: {abs_diff.mean():.4f})")
            axes[3].axis('off')
            fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = f"reports/baseline_visualizations/val_{basename}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Generated visualization for {fname} at {save_path}")

    # Compute Statistics for Approach A, B1, and B2
    df_a = pd.DataFrame(results_a)
    df_b1 = pd.DataFrame(results_b1)
    df_b2 = pd.DataFrame(results_b2)
    
    summary_rows = []
    
    for df, name in [(df_a, "A (Raw -> Bicubic -> Clip)"), 
                     (df_b1, "B1 (Clip -> Bicubic -> Clip-for-Metric)"), 
                     (df_b2, "B2 (Clip -> Bicubic -> Clip)")]:
        for col in ["PSNR", "SSIM", "MAE", "MSE"]:
            summary_rows.append({
                "Approach": name,
                "Metric": col,
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Std": df[col].std()
            })
            
    df_summary = pd.DataFrame(summary_rows)
    os.makedirs("reports", exist_ok=True)
    df_summary.to_csv("reports/baseline_results.csv", index=False)
    
    print("\nBaseline Results Summary:")
    print(df_summary.to_string(index=False))
    
    # Timing
    total_time_a = sum(times_a)
    avg_time_a = np.mean(times_a)
    print(f"\nInference Timing (Approach A):")
    print(f"  - Total validation inference time: {total_time_a:.2f} ms")
    print(f"  - Average time per image: {avg_time_a:.4f} ms")
    
    # Metric Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics_list = ["PSNR", "SSIM", "MAE", "MSE"]
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, metric in enumerate(metrics_list):
        axes[idx].hist(df_a[metric], bins=30, alpha=0.6, label="A (Clip After)", color=colors[0])
        axes[idx].hist(df_b2[metric], bins=30, alpha=0.6, label="B (Clip Before)", color=colors[1])
        axes[idx].set_title(f"{metric} Distribution")
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend()
        
    plt.tight_layout()
    dist_path = "reports/baseline_visualizations/metrics_distribution.png"
    plt.savefig(dist_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nGenerated metrics distribution histogram at {dist_path}")

if __name__ == "__main__":
    main()
