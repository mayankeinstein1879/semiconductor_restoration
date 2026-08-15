import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil

# Selected indices for visual analysis
SELECTED_INDICES = [0, 44, 88, 133, 177, 221, 266, 310, 354, 399]

def generate_visualizations():
    print("=== GENERATING FINAL RESTORATION VISUAL COMPARISONS ===")
    
    noisy_dir = "C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public/Test_NoisyLR/NoisyLR"
    restored_dir = "reports/final_test_outputs"
    vis_dir = "reports/final_test_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    brain_vis_dir = "C:/Users/Mayank Mukherjee/.gemini/antigravity/brain/9112b5e6-37c9-4876-af64-1ce37e4a2d0d/reports/final_test_visualizations"
    os.makedirs(brain_vis_dir, exist_ok=True)
    
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
    
    for idx in SELECTED_INDICES:
        if idx >= len(noisy_files):
            continue
            
        f_noisy = noisy_files[idx]
        basename = os.path.basename(f_noisy)
        f_restored = os.path.join(restored_dir, basename)
        
        if not os.path.exists(f_restored):
            print(f"Restored file not found: {f_restored}")
            continue
            
        # Load arrays
        lr = np.load(f_noisy).astype(np.float32)
        pred = np.load(f_restored).astype(np.float32)
        
        # Bicubic baseline
        bic = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Clip/clamp to [0,1] for display purposes
        lr_disp = np.clip(lr, 0, 1)
        bic_disp = np.clip(bic, 0, 1)
        pred_disp = np.clip(pred, 0, 1)
        
        # Plot 1: Full Comparison Grid
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(lr_disp, cmap="gray")
        axes[0].set_title(f"Degraded Input (128x128)\n{basename}")
        axes[0].axis("off")
        
        axes[1].imshow(bic_disp, cmap="gray")
        axes[1].set_title("Bicubic Upsampled (256x256)")
        axes[1].axis("off")
        
        axes[2].imshow(pred_disp, cmap="gray")
        axes[2].set_title("Final Restoration (256x256)")
        axes[2].axis("off")
        
        plt.tight_layout()
        plot_name = f"compare_final_{idx:06d}.png"
        plot_path = os.path.join(vis_dir, plot_name)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Copy to brain folder for embedding
        shutil.copy(plot_path, os.path.join(brain_vis_dir, plot_name))
        
        # Plot 2: Zoomed Crops Comparison
        h, w = pred.shape
        cy, cx = h // 2, w // 2
        crop_size = 32 # 64x64 region
        
        crop_lr = lr_disp[cy//2 - crop_size//2 : cy//2 + crop_size//2, cx//2 - crop_size//2 : cx//2 + crop_size//2]
        crop_bic = bic_disp[cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size]
        crop_pred = pred_disp[cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(crop_lr, cmap="gray", interpolation="nearest")
        axes[0].set_title("Input Crop (Zoomed)")
        axes[0].axis("off")
        
        axes[1].imshow(crop_bic, cmap="gray", interpolation="nearest")
        axes[1].set_title("Bicubic Crop (Zoomed)")
        axes[1].axis("off")
        
        axes[2].imshow(crop_pred, cmap="gray", interpolation="nearest")
        axes[2].set_title("Final Restored Crop (Zoomed)")
        axes[2].axis("off")
        
        plt.tight_layout()
        crop_plot_name = f"compare_final_{idx:06d}_crop.png"
        crop_plot_path = os.path.join(vis_dir, crop_plot_name)
        plt.savefig(crop_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Copy to brain folder for embedding
        shutil.copy(crop_plot_path, os.path.join(brain_vis_dir, crop_plot_name))
        
        print(f"Generated comparison plots for image index {idx:06d}")
        
    print("=== VISUAL COMPARISONS COMPLETED ===\n")

if __name__ == "__main__":
    generate_visualizations()
