import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block with two Conv2d layers, Batch Normalization, and LeakyReLU activation.
    Preserves spatial size and channel count.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act1(out + residual)

class DownsampleBlock(nn.Module):
    """
    Downsamples the feature map by 2x using a stride-2 Conv2d.
    Increases channel count from in_channels to out_channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpsampleBlock(nn.Module):
    """
    Upsamples the feature map by 2x using a stride-2 ConvTranspose2d.
    Decreases channel count from in_channels to out_channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))
