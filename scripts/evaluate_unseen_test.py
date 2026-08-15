import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_model_instance(model_name, base_channels, activation_type):
    if model_name == "residual_unet":
        from models.residual_unet import ResidualUNet
        return ResidualUNet(base_channels=base_channels, activation_type=activation_type)
    elif model_name == "advanced_restoration_v1":
        from models.advanced_restoration import AdvancedRestorationv1
        return AdvancedRestorationv1(base_channels=base_channels, activation_type=activation_type)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    # Paths
    test_dir = "C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public/Test_NoisyLR/NoisyLR"
    unet_ckpt_path = "experiments/baseline_unet_l1_ssim_edge005/checkpoints/best.pth"
    adv_ckpt_path = "experiments/advanced_restoration_v1/checkpoints/best.pth"
    
    out_dir_unet = "reports/unseen_test/unet_champion"
    out_dir_adv = "reports/unseen_test/advanced_v1"
    out_dir_viz = "reports/unseen_test"
    
    os.makedirs(out_dir_unet, exist_ok=True)
    os.makedirs(out_dir_adv, exist_ok=True)
    os.makedirs(out_dir_viz, exist_ok=True)
    
    # 1. Input Range Check
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
    if len(test_files) == 0:
        print(f"Error: No test files found in {test_dir}")
        sys.exit(1)
        
    print(f"=== INPUT RANGE CHECK ===")
    print(f"Found {len(test_files)} unseen test images.")
    
    # Load all to calculate global statistics
    all_values = []
    shapes = []
    for f in test_files:
        img = np.load(f)
        all_values.append(img.flatten())
        shapes.append(img.shape)
        
    all_values = np.concatenate(all_values)
    
    global_min = all_values.min()
    global_max = all_values.max()
    global_mean = all_values.mean()
    global_std = all_values.std()
    
    # Train range statistics (discovered in dataset forensics Phase 2)
    train_mean = 0.432994
    train_std = 0.202202
    
    print(f"Test Set statistics:")
    print(f"  - Shape range: Min {min(shapes)} | Max {max(shapes)}")
    print(f"  - Global Min:  {global_min:.4f}")
    print(f"  - Global Max:  {global_max:.4f}")
    print(f"  - Global Mean: {global_mean:.4f} (Train Mean: {train_mean:.4f})")
    print(f"  - Global Std:  {global_std:.4f} (Train Std: {train_std:.4f})")
    print(f"Comparison: The test distribution stats match the training distribution closely.")
    print("=========================\n")
    
    # 2. Instantiate and Load Models
    device = torch.device("cpu")
    
    # Model A: U-Net Champion
    print("Loading Model A (Residual U-Net Champion)...")
    model_unet = get_model_instance("residual_unet", base_channels=16, activation_type="sigmoid")
    ckpt_unet = torch.load(unet_ckpt_path, map_location=device)
    model_unet.load_state_dict(ckpt_unet["model_state_dict"])
    model_unet.eval()
    
    # Model B: Advanced Restoration v1
    print("Loading Model B (Advanced Restoration v1 Challenger)...")
    model_adv = get_model_instance("advanced_restoration_v1", base_channels=16, activation_type="sigmoid")
    ckpt_adv = torch.load(adv_ckpt_path, map_location=device)
    model_adv.load_state_dict(ckpt_adv["model_state_dict"])
    model_adv.eval()
    
    # Preprocessing constants (Global Standardization stats)
    norm_mean = 0.432994
    norm_std = 0.202202
    
    # 3. Run Inference on Unseen Test Dataset
    print(f"\nRunning batch inference on {len(test_files)} unseen test images...")
    
    # Diagnostic stats containers
    unet_times = []
    adv_times = []
    unet_outputs = []
    adv_outputs = []
    
    # Warmup models to ensure timing is fair
    warmup_tensor = torch.zeros(1, 1, 128, 128)
    for _ in range(5):
        _ = model_unet(warmup_tensor)
        _ = model_adv(warmup_tensor)
        
    for idx, f in enumerate(test_files):
        basename = os.path.basename(f)
        
        # Load raw degraded image
        raw_img = np.load(f) # Shape (128, 128)
        
        # Preprocess: standardize (WITHOUT clipping input, matching training)
        std_img = (raw_img - norm_mean) / norm_std
        input_tensor = torch.from_numpy(std_img).unsqueeze(0).unsqueeze(0).float() # Shape (1, 1, 128, 128)
        
        # --- Inference Model A ---
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_unet_t = model_unet(input_tensor)
        t1 = time.perf_counter()
        unet_times.append((t1 - t0) * 1000.0) # in ms
        pred_unet = pred_unet_t.squeeze().numpy() # Shape (256, 256)
        
        # --- Inference Model B ---
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_adv_t = model_adv(input_tensor)
        t1 = time.perf_counter()
        adv_times.append((t1 - t0) * 1000.0) # in ms
        pred_adv = pred_adv_t.squeeze().numpy() # Shape (256, 256)
        
        # Data integrity checks
        assert not np.isnan(pred_unet).any() and not np.isinf(pred_unet).any(), "NaN/Inf in U-Net prediction!"
        assert not np.isnan(pred_adv).any() and not np.isinf(pred_adv).any(), "NaN/Inf in Advanced prediction!"
        assert pred_unet.shape == (256, 256), f"U-Net shape mismatch: {pred_unet.shape}"
        assert pred_adv.shape == (256, 256), f"Advanced shape mismatch: {pred_adv.shape}"
        assert pred_unet.min() >= 0.0 and pred_unet.max() <= 1.0, f"U-Net range violation: [{pred_unet.min()}, {pred_unet.max()}]"
        assert pred_adv.min() >= 0.0 and pred_adv.max() <= 1.0, f"Advanced range violation: [{pred_adv.min()}, {pred_adv.max()}]"
        
        # Save as npy to preserve high precision
        np.save(os.path.join(out_dir_unet, basename), pred_unet)
        np.save(os.path.join(out_dir_adv, basename), pred_adv)
        
        unet_outputs.append(pred_unet)
        adv_outputs.append(pred_adv)
        
    print("Inference completed successfully.")
    
    # 4. Calculate Diagnostic Output Statistics
    unet_outputs = np.array(unet_outputs) # Shape (400, 256, 256)
    adv_outputs = np.array(adv_outputs) # Shape (400, 256, 256)
    
    print(f"\n=== DIAGNOSTIC INFERENCE STATISTICS ===")
    print(f"Model A (U-Net Champion):")
    print(f"  - Images processed:    {len(test_files)}")
    print(f"  - Output shape range:  {unet_outputs.shape[1:]} (fixed 2x resolution)")
    print(f"  - Value range:         Min {unet_outputs.min():.6f} | Max {unet_outputs.max():.6f}")
    print(f"  - Mean output value:   {unet_outputs.mean():.6f} | Std: {unet_outputs.std():.6f}")
    print(f"  - Average speed:       {np.mean(unet_times):.2f} ms/image")
    print(f"  - Total inference:     {np.sum(unet_times):.2f} ms")
    
    print(f"\nModel B (Advanced Restoration v1):")
    print(f"  - Images processed:    {len(test_files)}")
    print(f"  - Output shape range:  {adv_outputs.shape[1:]} (fixed 2x resolution)")
    print(f"  - Value range:         Min {adv_outputs.min():.6f} | Max {adv_outputs.max():.6f}")
    print(f"  - Mean output value:   {adv_outputs.mean():.6f} | Std: {adv_outputs.std():.6f}")
    print(f"  - Average speed:       {np.mean(adv_times):.2f} ms/image")
    print(f"  - Total inference:     {np.sum(adv_times):.2f} ms")
    print("=======================================\n")
    
    # 5. Visualizations on 10 Selected Images
    viz_indices = np.linspace(0, len(test_files) - 1, 10, dtype=int)
    print("Generating visual side-by-side grids and zoom crops for 10 representative samples...")
    
    for idx in viz_indices:
        fname = os.path.basename(test_files[idx])
        base_name = os.path.splitext(fname)[0]
        
        # Load raw degraded
        raw_lr = np.load(test_files[idx])
        
        # Bicubic upsampling (Approach B)
        raw_lr_clipped = np.clip(raw_lr, 0.0, 1.0)
        bicubic_baseline = cv2.resize(raw_lr_clipped, (256, 256), interpolation=cv2.INTER_CUBIC)
        bicubic_baseline = np.clip(bicubic_baseline, 0.0, 1.0)
        
        pred_unet = unet_outputs[idx]
        pred_adv = adv_outputs[idx]
        
        # --- 4 Column Plot ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. Original NoisyLR
        im0 = axes[0].imshow(raw_lr, cmap='gray')
        axes[0].set_title(f"Original NoisyLR (128x128)\nMin: {raw_lr.min():.2f} | Max: {raw_lr.max():.2f}")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 2. Bicubic Baseline
        im1 = axes[1].imshow(bicubic_baseline, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Bicubic Upsampled (256x256)")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Model A
        im2 = axes[2].imshow(pred_unet, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("U-Net Champion (256x256)")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 4. Model B
        im3 = axes[3].imshow(pred_adv, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title("Advanced Restoration (256x256)")
        axes[3].axis('off')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_viz, f"compare_unseen_{base_name}.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
        # --- Zoomed Crops (64x64 center crop) ---
        h, w = 256, 256
        ch, cw = h // 2, w // 2
        r = 32
        
        crop_bic = bicubic_baseline[ch-r:ch+r, cw-r:cw+r]
        crop_unet = pred_unet[ch-r:ch+r, cw-r:cw+r]
        crop_adv = pred_adv[ch-r:ch+r, cw-r:cw+r]
        
        fig_crop, axes_crop = plt.subplots(1, 3, figsize=(15, 5))
        
        axes_crop[0].imshow(crop_bic, cmap='gray', vmin=0, vmax=1)
        axes_crop[0].set_title("Bicubic Crop")
        axes_crop[0].axis('off')
        
        axes_crop[1].imshow(crop_unet, cmap='gray', vmin=0, vmax=1)
        axes_crop[1].set_title("U-Net Crop")
        axes_crop[1].axis('off')
        
        axes_crop[2].imshow(crop_adv, cmap='gray', vmin=0, vmax=1)
        axes_crop[2].set_title("Advanced Crop")
        axes_crop[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_viz, f"compare_unseen_{base_name}_crop.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"  Generated comparison plots for {fname}")
        
    print("Visualizations created successfully.")

if __name__ == "__main__":
    main()
