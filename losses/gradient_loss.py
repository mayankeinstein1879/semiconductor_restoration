import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLoss(nn.Module):
    """
    Edge-aware Loss using Sobel Filters.
    Computes L1 difference between the horizontal (Gx) and vertical (Gy) 
    gradients of the prediction and target.
    """
    def __init__(self):
        super().__init__()
        # Define Sobel Kernels
        kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 3, 3)
        
        kernel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 3, 3)
        
        # Register buffers so they move to the correct device but are not trainable
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        
    def forward(self, pred, target):
        # Enforce correct device
        kx = self.kernel_x.to(pred.device)
        ky = self.kernel_y.to(pred.device)
        
        # Check channels and replicate kernels if channel count > 1
        channels = pred.size(1)
        if channels > 1:
            kx = kx.repeat(channels, 1, 1, 1)
            ky = ky.repeat(channels, 1, 1, 1)
            
        # Compute horizontal and vertical gradients
        # Use padding=1 to preserve spatial size
        grad_x_pred = F.conv2d(pred, kx, padding=1, groups=channels)
        grad_y_pred = F.conv2d(pred, ky, padding=1, groups=channels)
        
        grad_x_gt = F.conv2d(target, kx, padding=1, groups=channels)
        grad_y_gt = F.conv2d(target, ky, padding=1, groups=channels)
        
        # L1-type difference
        loss_x = torch.mean(torch.abs(grad_x_pred - grad_x_gt))
        loss_y = torch.mean(torch.abs(grad_y_pred - grad_y_gt))
        
        return loss_x + loss_y
