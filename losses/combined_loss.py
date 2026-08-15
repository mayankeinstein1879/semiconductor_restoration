import torch
import torch.nn as nn
from losses.ssim_loss import SSIMLoss
from losses.gradient_loss import SobelLoss

class CombinedLoss(nn.Module):
    """
    Combined Loss class supporting L1 loss, SSIM loss, and Sobel gradient loss.
    L = l1_weight * L1 + ssim_weight * L_ssim + edge_weight * L_edge
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.10, edge_weight=0.05):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = SobelLoss()
        
    def forward(self, pred, target):
        loss_val = 0.0
        
        if self.l1_weight > 0:
            l1_val = self.l1_loss(pred, target)
            loss_val += self.l1_weight * l1_val
            
        if self.ssim_weight > 0:
            ssim_val = self.ssim_loss(pred, target)
            loss_val += self.ssim_weight * ssim_val
            
        if self.edge_weight > 0:
            edge_val = self.edge_loss(pred, target)
            loss_val += self.edge_weight * edge_val
            
        return loss_val
