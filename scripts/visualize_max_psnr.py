import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import get_model

def main():
    print("=== GENERATING MAX PSNR IMAGE VISUAL COMPARISON ===")
    
    # Paths
    noisy_path = "C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public/train/train/NoisyLR/001095.npy"
    gt_path = "C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public/train/train/GT/001095.npy"
    checkpoint_path = "checkpoints/final_model.pth"
    
    brain_vis_dir = "C:/Users/Mayank Mukherjee/.gemini/antigravity/brain/9112b5e6-37c9-4876-af64-1ce37e4a2d0d/reports"
    os.makedirs(brain_vis_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = get_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Load stats
    norm_stats = ckpt.get("normalization", {"mean": 0.432994, "std": 0.202202})
    norm_mean = norm_stats["mean"]
    norm_std = norm_stats["std"]
    
    # Load images
    lr = np.load(noisy_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)
    
    # Run inference
    std_lr = (lr - norm_mean) / norm_std
    input_t = torch.from_numpy(std_lr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_t).squeeze().cpu().numpy()
        
    # Clip displays
    lr_disp = np.clip(lr, 0, 1)
    gt_disp = np.clip(gt, 0, 1)
    pred_disp = np.clip(pred, 0, 1)
    
    # Plot 1: Full comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr_disp, cmap="gray")
    axes[0].set_title("Degraded Input (128x128)\n001095.npy")
    axes[0].axis("off")
    
    axes[1].imshow(gt_disp, cmap="gray")
    axes[1].set_title("Clean Ground Truth (256x256)")
    axes[1].axis("off")
    
    axes[2].imshow(pred_disp, cmap="gray")
    axes[2].set_title("Final Restored (256x256)\nPSNR: 37.58 dB")
    axes[2].axis("off")
    
    plt.tight_layout()
    full_plot_path = os.path.join(brain_vis_dir, "compare_max_psnr_001095.png")
    plt.savefig(full_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Center crop zoom comparison
    h, w = pred.shape
    cy, cx = h // 2, w // 2
    crop_size = 32 # 64x64 region on output, 32x32 on input
    
    crop_lr = lr_disp[cy//2 - crop_size//2 : cy//2 + crop_size//2, cx//2 - crop_size//2 : cx//2 + crop_size//2]
    crop_gt = gt_disp[cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size]
    crop_pred = pred_disp[cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(crop_lr, cmap="gray", interpolation="nearest")
    axes[0].set_title("Input Crop (Zoomed)")
    axes[0].axis("off")
    
    axes[1].imshow(crop_gt, cmap="gray", interpolation="nearest")
    axes[1].set_title("Ground Truth Crop (Zoomed)")
    axes[1].axis("off")
    
    axes[2].imshow(crop_pred, cmap="gray", interpolation="nearest")
    axes[2].set_title("Restored Crop (Zoomed)")
    axes[2].axis("off")
    
    plt.tight_layout()
    crop_plot_path = os.path.join(brain_vis_dir, "compare_max_psnr_001095_crop.png")
    plt.savefig(crop_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Generated visual comparisons for 001095.npy.")

if __name__ == "__main__":
    main()
