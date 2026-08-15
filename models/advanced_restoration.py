import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import DownsampleBlock, UpsampleBlock

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D image tensors.
    Optimized via permuting to channel-last layout for contiguous F.layer_norm execution.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        
    def forward(self, x):
        # Permute to (B, H, W, C) for contiguous layout
        x = x.permute(0, 2, 3, 1)
        # Squeeze parameters to 1D for F.layer_norm compatibility
        w = self.weight.squeeze()
        b = self.bias.squeeze()
        x = F.layer_norm(x, (x.size(-1),), w, b, self.eps)
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x

class TransposedAttention(nn.Module):
    """
    Multi-Dilation Transposed Attention (MDTA).
    Computes self-attention across the channel dimension rather than spatial.
    Complexity is linear in spatial resolution O(C^2) rather than quadratic.
    """
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Q, K, V projections using 1x1 conv followed by 3x3 depthwise conv
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dw = nn.Conv2d(
            dim * 3, dim * 3, 
            kernel_size=3, stride=1, padding=1, 
            groups=dim * 3, bias=bias
        )
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape to (B, heads, C/heads, H*W)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        # L2 normalize along the spatial dimension (HW)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Transposed attention across channels: (C/heads) x (C/heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # Apply attention weight to values
        out = attn @ v
        
        # Reshape back to original spatial feature map
        out = out.reshape(b, c, h, w)
        
        # Final output projection
        out = self.project_out(out)
        return out

class GatedDilationFFN(nn.Module):
    """
    Gated Dilation Feed-Forward Network (GDFN).
    Uses a gating mechanism (GELU path multiplied by linear path)
    with depthwise convolutions for local information filtering.
    """
    def __init__(self, dim, expansion_factor=2.0, bias=False):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        
        # Project to expanded space (2x hidden features for gating split)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 
            kernel_size=3, stride=1, padding=1, 
            groups=hidden_features * 2, bias=bias
        )
        
        # Project back to original feature dimension
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        # Project and split into parallel paths
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        
        # Gate operation (GELU * linear)
        gated = F.gelu(x1) * x2
        
        # Project out
        return self.project_out(gated)

class RestormerBlock(nn.Module):
    """
    Restormer Block combining Transposed Attention and Gated Dilation FFN.
    """
    def __init__(self, dim, num_heads, ffn_expansion=2.0, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = TransposedAttention(dim, num_heads, bias)
        
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDilationFFN(dim, ffn_expansion, bias)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class AdvancedRestorationv1(nn.Module):
    """
    Advanced lightweight restoration architecture (Restormer-inspired).
    Integrates Restormer Blocks into a multi-scale encoder-decoder skip network.
    Input: (B, 1, 128, 128)
    Output: (B, 1, 256, 256)
    """
    def __init__(self, base_channels=16, activation_type="sigmoid"):
        super().__init__()
        self.activation_type = activation_type.lower()
        
        # --- Encoder ---
        self.in_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False)
        self.enc_block0 = RestormerBlock(base_channels, num_heads=1)
        
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.enc_block1 = RestormerBlock(base_channels * 2, num_heads=2)
        
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        
        # --- Bottleneck ---
        self.bottleneck = RestormerBlock(base_channels * 4, num_heads=4)
        
        # --- Decoder ---
        self.up1 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.dec_conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False)
        self.dec_block1 = RestormerBlock(base_channels * 2, num_heads=2)
        
        self.up2 = UpsampleBlock(base_channels * 2, base_channels)
        self.dec_conv2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False)
        self.dec_block2 = RestormerBlock(base_channels, num_heads=1)
        
        # --- Final 2x learned upsampling path ---
        self.final_up = UpsampleBlock(base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        # 1. Bicubic skip connection
        bicubic_upsampled = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        
        # 2. Advanced Transformer Feature Extraction
        s0 = self.in_conv(x)
        s0 = self.enc_block0(s0)
        
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
        
        # 2x learned upsampling of residual features
        res_map = self.final_up(d0)
        res_map = self.out_conv(res_map)
        
        # Combine
        combined = bicubic_upsampled + res_map
        
        # Apply output constraint
        if self.activation_type == "sigmoid":
            return torch.sigmoid(combined)
        elif self.activation_type == "clamp":
            return torch.clamp(combined, 0.0, 1.0)
        else:
            return combined
