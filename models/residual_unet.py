import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import ResidualBlock, DownsampleBlock, UpsampleBlock

class ResidualUNet(nn.Module):
    """
    Lightweight Residual U-Net for joint denoising and 2x super-resolution.
    Input: (B, 1, 128, 128)
    Output: (B, 1, 256, 256)
    """
    def __init__(self, base_channels=32, activation_type="sigmoid"):
        super().__init__()
        self.activation_type = activation_type.lower()
        
        # --- Encoder ---
        self.in_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False)
        self.enc_block0 = ResidualBlock(base_channels)
        
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.enc_block1 = ResidualBlock(base_channels * 2)
        
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        
        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(base_channels * 4)
        
        # --- Decoder ---
        self.up1 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.dec_conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False)
        self.dec_block1 = ResidualBlock(base_channels * 2)
        
        self.up2 = UpsampleBlock(base_channels * 2, base_channels)
        self.dec_conv2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False)
        self.dec_block2 = ResidualBlock(base_channels)
        
        # --- Final 2x learned upsampling ---
        self.final_up = UpsampleBlock(base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        # 1. Bicubic path
        # Note: bicubic interpolation of input tensor (B, 1, 128, 128) -> (B, 1, 256, 256)
        bicubic_upsampled = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        
        # 2. UNet path (residual map)
        s0 = self.act_and_bn_in(x)
        
        # Down 1
        s1 = self.down1(s0)
        s1 = self.enc_block1(s1)
        
        # Bottleneck
        b = self.down2(s1)
        b = self.bottleneck(b)
        
        # Up 1
        d1 = self.up1(b)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec_conv1(d1)
        d1 = self.dec_block1(d1)
        
        # Up 2
        d0 = self.up2(d1)
        d0 = torch.cat([d0, s0], dim=1)
        d0 = self.dec_conv2(d0)
        d0 = self.dec_block2(d0)
        
        # Final learned 2x upsampling
        res_map = self.final_up(d0)
        res_map = self.out_conv(res_map)
        
        # Combine
        combined = bicubic_upsampled + res_map
        
        # Apply output constraints
        if self.activation_type == "sigmoid":
            return torch.sigmoid(combined)
        elif self.activation_type == "clamp":
            return torch.clamp(combined, 0.0, 1.0)
        else:
            return combined

    def act_and_bn_in(self, x):
        return self.enc_block0(self.in_conv(x))
