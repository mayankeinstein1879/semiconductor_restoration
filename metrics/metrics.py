import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(gt, pred, data_range=1.0):
    """Computes Peak Signal-to-Noise Ratio."""
    # Ensure values are within reasonable bounds and check for exact match
    mse_val = np.mean((gt - pred) ** 2)
    if mse_val == 0:
        return 100.0
    return float(peak_signal_noise_ratio(gt, pred, data_range=data_range))

def compute_ssim(gt, pred, data_range=1.0):
    """Computes Structural Similarity Index."""
    return float(structural_similarity(gt, pred, data_range=data_range, channel_axis=None))

def compute_mae(gt, pred):
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(gt - pred)))

def compute_mse(gt, pred):
    """Computes Mean Squared Error."""
    return float(np.mean((gt - pred) ** 2))

def compute_all_metrics(gt, pred, data_range=1.0):
    """Computes all evaluation metrics between GT and prediction."""
    # Ensure they are floating point arrays and clipped appropriately if needed
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    
    return {
        "PSNR": compute_psnr(gt, pred, data_range=data_range),
        "SSIM": compute_ssim(gt, pred, data_range=data_range),
        "MAE": compute_mae(gt, pred),
        "MSE": compute_mse(gt, pred)
    }
