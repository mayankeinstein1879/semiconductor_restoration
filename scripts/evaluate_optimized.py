import os
import sys
import time
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    config_path = "configs/advanced_restoration_v1.yaml"
    checkpoint_path = "experiments/advanced_restoration_v1/checkpoints/best.pth"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    
    # Load model
    print("Loading optimized advanced model...")
    model = get_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Load dataset
    dataset = SemiconductorDataset(
        dataset_root=config["dataset"]["root"],
        split_file=config["dataset"]["val_split"],
        mode="val",
        normalize=config["dataset"]["normalize"],
        global_mean=config["dataset"]["global_mean"],
        global_std=config["dataset"]["global_std"],
        augment=False
    )
    
    # Inference and timing
    print("Evaluating on 320 canonical validation images...")
    psnrs, ssims, maes, mses, grad_maes = [], [], [], [], []
    inference_times = []
    
    # Warmup
    warmup_tensor = torch.zeros(1, 1, 128, 128)
    for _ in range(10):
        _ = model(warmup_tensor)
        
    for idx in range(len(dataset)):
        lr_tensor, gt_tensor = dataset[idx]
        lr_tensor = lr_tensor.unsqueeze(0)
        gt_np = gt_tensor.squeeze().numpy()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(lr_tensor).squeeze().numpy()
        t1 = time.perf_counter()
        
        inference_times.append((t1 - t0) * 1000.0) # ms
        
        # Calculate metrics
        m = compute_all_metrics(gt_np, pred)
        psnrs.append(m["PSNR"])
        ssims.append(m["SSIM"])
        maes.append(m["MAE"])
        mses.append(m["MSE"])
        
        # Sobel gradients
        gx_gt, gy_gt = get_sobel_gradients(gt_np, device)
        gx_pred, gy_pred = get_sobel_gradients(pred, device)
        grad_mae = np.mean(np.abs(gx_pred - gx_gt) + np.abs(gy_pred - gy_gt))
        grad_maes.append(grad_mae)
        
    print("\n=== OPTIMIZED MODEL VALIDATION RESULTS ===")
    print(f"PSNR Mean:      {np.mean(psnrs):.4f} dB | Median: {np.median(psnrs):.4f} dB")
    print(f"SSIM Mean:      {np.mean(ssims):.4f} | Median: {np.median(ssims):.4f}")
    print(f"MAE Mean:       {np.mean(maes):.6f}")
    print(f"MSE Mean:       {np.mean(mses):.6f}")
    print(f"Gradient MAE:   {np.mean(grad_maes):.6f}")
    
    print("\n=== OPTIMIZED INFERENCE SPEED ===")
    print(f"Total time:     {np.sum(inference_times):.2f} ms")
    print(f"Average speed:  {np.mean(inference_times):.2f} ms/image")
    print(f"Throughput:     {1000.0 / np.mean(inference_times):.2f} images/sec (FPS)")
    print("==========================================\n")

if __name__ == "__main__":
    main()
