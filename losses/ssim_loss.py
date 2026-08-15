import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    """Creates a 2D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Calculates SSIM between two images."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.0001
    C2 = 0.0009

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])

class SSIMLoss(nn.Module):
    """
    SSIM Loss function.
    Computes: L_ssim = 1 - SSIM(x, y)
    Suitable for single channel (grayscale) images.
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, img1, img2):
        # Transfer window buffer to match input device if needed
        # (register_buffer handles device transfer when moving the module, 
        # but check for safety)
        _, channel, _, _ = img1.size()
        
        if channel != self.channel:
            # Recreate window if channel mismatch
            window = create_window(self.window_size, channel).to(img1.device)
        else:
            window = self.window.to(img1.device)
            
        ssim_val = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1.0 - ssim_val
