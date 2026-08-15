import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from data.dataset import SemiconductorDataset
from models.factory import get_model
from metrics.metrics import compute_all_metrics

def main():
    config_path = "configs/baseline_unet_16.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    
    # 1. Load checkpoints
    # Model L1
    model_l1 = get_model(config)
    ckpt_l1 = torch.load("experiments/baseline_unet_16/checkpoints/best.pth", map_location=device)
    model_l1.load_state_dict(ckpt_l1["model_state_dict"])
    model_l1.eval()
    
    # Model L1+SSIM
    config_ssim = config.copy()
    config_ssim["loss"]["ssim_weight"] = 0.10
    model_ssim = get_model(config_ssim)
    ckpt_ssim = torch.load("experiments/baseline_unet_l1_ssim/checkpoints/best.pth", map_location=device)
    model_ssim.load_state_dict(ckpt_ssim["model_state_dict"])
    model_ssim.eval()
    
    # Model L1+SSIM+Edge (Previous Champion)
    config_edge = config.copy()
    config_edge["loss"]["ssim_weight"] = 0.10
    config_edge["loss"]["edge_weight"] = 0.05
    model_edge = get_model(config_edge)
    ckpt_edge = torch.load("experiments/baseline_unet_l1_ssim_edge005/checkpoints/best.pth", map_location=device)
    model_edge.load_state_dict(ckpt_edge["model_state_dict"])
    model_edge.eval()
    
    # Model Advanced Restoration (Challenger)
    config_adv = config.copy()
    config_adv["model"]["name"] = "advanced_restoration_v1"
    config_adv["loss"]["ssim_weight"] = 0.10
    config_adv["loss"]["edge_weight"] = 0.05
    model_adv = get_model(config_adv)
    ckpt_adv = torch.load("experiments/advanced_restoration_v1/checkpoints/best.pth", map_location=device)
    model_adv.load_state_dict(ckpt_adv["model_state_dict"])
    model_adv.eval()
    
    # 2. Setup dataset
    dataset = SemiconductorDataset(
        dataset_root=config["dataset"]["root"],
        split_file=config["dataset"]["val_split"],
        mode="val",
        normalize=config["dataset"]["normalize"],
        global_mean=config["dataset"]["global_mean"],
        global_std=config["dataset"]["global_std"],
        augment=False
    )
    
    # Select same 10 filenames
    val_filenames = dataset.filenames
    viz_indices = np.linspace(0, len(val_filenames)-1, 10, dtype=int)
    
    save_dir = "reports/model_comparisons"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visual comparisons including Advanced Restoration model...")
    for idx in viz_indices:
        fname = val_filenames[idx]
        basename = os.path.splitext(fname)[0]
        
        lr_tensor, gt_tensor = dataset[idx]
        lr_tensor = lr_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred_l1 = model_l1(lr_tensor).squeeze().numpy()
            pred_ssim = model_ssim(lr_tensor).squeeze().numpy()
            pred_edge = model_edge(lr_tensor).squeeze().numpy()
            pred_adv = model_adv(lr_tensor).squeeze().numpy()
            
        gt_np = gt_tensor.squeeze().numpy()
        
        # Load raw degraded
        raw_lr_path = os.path.join(config["dataset"]["root"], "train", "train", "NoisyLR", fname)
        raw_lr = np.load(raw_lr_path)
        
        # Bicubic upsampling (Approach B)
        raw_lr_clipped = np.clip(raw_lr, 0.0, 1.0)
        bicubic_baseline = cv2.resize(raw_lr_clipped, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        bicubic_baseline = np.clip(bicubic_baseline, 0.0, 1.0)
        
        # Compute individual metrics
        m_bic = compute_all_metrics(gt_np, bicubic_baseline)
        m_l1 = compute_all_metrics(gt_np, pred_l1)
        m_ssim = compute_all_metrics(gt_np, pred_ssim)
        m_edge = compute_all_metrics(gt_np, pred_edge)
        m_adv = compute_all_metrics(gt_np, pred_adv)
        
        # --- 7 Column Plot ---
        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        
        # 1. Degraded
        im0 = axes[0].imshow(raw_lr, cmap='gray')
        axes[0].set_title(f"Degraded 128x128\nMin: {raw_lr.min():.2f} | Max: {raw_lr.max():.2f}")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 2. Bicubic
        im1 = axes[1].imshow(bicubic_baseline, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Bicubic Baseline\nPSNR: {m_bic['PSNR']:.2f} dB")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. L1 U-Net
        im2 = axes[2].imshow(pred_l1, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"L1 U-Net\nPSNR: {m_l1['PSNR']:.2f} dB")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 4. L1+SSIM U-Net
        im3 = axes[3].imshow(pred_ssim, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f"L1+SSIM U-Net\nPSNR: {m_ssim['PSNR']:.2f} dB")
        axes[3].axis('off')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        # 5. L1+SSIM+Edge U-Net
        im4 = axes[4].imshow(pred_edge, cmap='gray', vmin=0, vmax=1)
        axes[4].set_title(f"L1+SSIM+Edge\nPSNR: {m_edge['PSNR']:.2f} dB")
        axes[4].axis('off')
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
        # 6. Advanced Restoration U-Net
        im5 = axes[5].imshow(pred_adv, cmap='gray', vmin=0, vmax=1)
        axes[5].set_title(f"Advanced Rest.\nPSNR: {m_adv['PSNR']:.2f} dB")
        axes[5].axis('off')
        fig.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
        
        # 7. Ground Truth
        im6 = axes[6].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[6].set_title(f"Ground Truth 256x256")
        axes[6].axis('off')
        fig.colorbar(im6, ax=axes[6], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"compare_{basename}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # --- Zoomed Crops Plot (64x64 region in the center) ---
        h, w = gt_np.shape
        ch, cw = h // 2, w // 2
        r = 32
        
        crop_bic = bicubic_baseline[ch-r:ch+r, cw-r:cw+r]
        crop_l1 = pred_l1[ch-r:ch+r, cw-r:cw+r]
        crop_ssim = pred_ssim[ch-r:ch+r, cw-r:cw+r]
        crop_edge = pred_edge[ch-r:ch+r, cw-r:cw+r]
        crop_adv = pred_adv[ch-r:ch+r, cw-r:cw+r]
        crop_gt = gt_np[ch-r:ch+r, cw-r:cw+r]
        
        fig_crop, axes_crop = plt.subplots(1, 6, figsize=(24, 4))
        
        axes_crop[0].imshow(crop_bic, cmap='gray', vmin=0, vmax=1)
        axes_crop[0].set_title("Bicubic Crop")
        axes_crop[0].axis('off')
        
        axes_crop[1].imshow(crop_l1, cmap='gray', vmin=0, vmax=1)
        axes_crop[1].set_title("L1 Crop")
        axes_crop[1].axis('off')
        
        axes_crop[2].imshow(crop_ssim, cmap='gray', vmin=0, vmax=1)
        axes_crop[2].set_title("L1+SSIM Crop")
        axes_crop[2].axis('off')
        
        axes_crop[3].imshow(crop_edge, cmap='gray', vmin=0, vmax=1)
        axes_crop[3].set_title("L1+SSIM+Edge Crop")
        axes_crop[3].axis('off')
        
        axes_crop[4].imshow(crop_adv, cmap='gray', vmin=0, vmax=1)
        axes_crop[4].set_title("Advanced Crop")
        axes_crop[4].axis('off')
        
        axes_crop[5].imshow(crop_gt, cmap='gray', vmin=0, vmax=1)
        axes_crop[5].set_title("GT Crop")
        axes_crop[5].axis('off')
        
        plt.tight_layout()
        save_crop_path = os.path.join(save_dir, f"compare_{basename}_crop.png")
        plt.savefig(save_crop_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"  Generated comparison and zoom crop for {fname}")

if __name__ == "__main__":
    main()
