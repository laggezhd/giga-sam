import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """
    Lightweight Attention Mechanism.
    Helps the model focus on global features with minimal parameter cost.

    This SE block takes as input a feature map of shape (B, C, H, W).
    It outputs a feature map of the same shape after applying channel-wise attention.

    The reduction ratio controls the bottleneck in the SE block. In the original paper,
    the authors used a default reduction of 16. Smaller reduction ratio increases the 
    number of parameters! --> should retrain at 16.

    Hardsigmoid is used since it is faster on MCU than Sigmoid. --> check if supported on MCU
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        # Reduced dimension must be at least 1
        reduced_dim = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # BxCx1x1
            nn.Conv2d(in_channels, reduced_dim, 1),  # Bx(reduced_dim)x1x1
            nn.ReLU(inplace=True),                   # Bx(reduced_dim)x1x1
            nn.Conv2d(reduced_dim, in_channels, 1),  # BxCx1x1
            nn.Hardsigmoid()                         # BxCx1x1              
        )

    def forward(self, x):
        return x * self.fc(x)  # channel-wise multiplication (attention)


class InvertedResidual(nn.Module):
    """
    MobileNetV2/V3 style block: Expand -> Depthwise -> Project

    We use ReLU6 activations for better quantization compatibility.
    """
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, use_se=True):
        super().__init__()
        hidden_dim = int(round(in_c * expand_ratio))
        self.use_res_connect = (stride == 1 and in_c == out_c)

        layers = []
        # 1. Expansion Phase (Pointwise 1x1)    
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True)) 

        # 2. Depthwise Convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 3. Squeeze and Excite (Optional but recommended)
        if use_se:
            layers.append(SqueezeExcite(hidden_dim))

        # 4. Projection Phase (Pointwise 1x1)
        layers.append(nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MicroSAM(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # Initial stem: Bx3x64x64 -> Bx16x64x64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True) # good for quantization
        )

        # Encoder (Downsampling path)
        # Stage 1: Bx16x64x64 -> Bx16x64x64
        self.enc1 = InvertedResidual(16, 16, stride=1, expand_ratio=2) 
        
        # Stage 2: 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=4), 
            InvertedResidual(24, 24, stride=1, expand_ratio=4)
        )

        # Stage 3: 32x32 -> 16x16
        self.enc3 = nn.Sequential(
            InvertedResidual(24, 48, stride=2, expand_ratio=4),
            InvertedResidual(48, 48, stride=1, expand_ratio=4)
        )
        
        # Stage 4: 16x16 -> 8x8 (Bottleneck)
        self.enc4 = nn.Sequential(
            InvertedResidual(48, 96, stride=2, expand_ratio=4),
            InvertedResidual(96, 96, stride=1, expand_ratio=4)
        )

        # Decoder (Upsampling path) - Using Bilinear or Nearest
        # Note: We use simple Conv blocks after upsampling to save params vs InvertedResiduals
        
        # Up 1: 8x8 -> 16x16
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce4 = nn.Conv2d(96, 48, 1, bias=False) # Reduce bottleneck channels
        self.dec1 = nn.Sequential(
            nn.Conv2d(48+48, 48, 3, padding=1, groups=48, bias=False), # Depthwise
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True),
            nn.Conv2d(48, 48, 1, bias=False), # Pointwise
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True),
        )

        # Up 2: 16x16 -> 32x32
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce3 = nn.Conv2d(48, 24, 1, bias=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(24+24, 24, 3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            nn.Conv2d(24, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
        )

        # Up 3: 32x32 -> 64x64
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce2 = nn.Conv2d(24, 16, 1, bias=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(16+16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # Head
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)       # Bx16x64x64
        x1 = self.enc1(x1)      # Bx16x64x64
        x2 = self.enc2(x1)      # Bx24x32x32
        x3 = self.enc3(x2)      # Bx48x16x16
        x4 = self.enc4(x3)      # Bx96x8x8

        # Decoder with Skip Connections
        # Concatenation usually yields better IoU than addition for segmentation
        
        # Block 1
        d1 = self.up1(x4)               # Bx96x16x16
        d1 = self.reduce4(d1)           # Bx48x16x16
        d1 = torch.cat([d1, x3], dim=1) # Bx96x16x16 --> C: 48+48
        d1 = self.dec1(d1)              # Bx48x16x16
        
        # Block 2
        d2 = self.up2(d1)               # Bx48x32x32
        d2 = self.reduce3(d2)           # Bx24x32x32
        d2 = torch.cat([d2, x2], dim=1) # Bx48x32x32 --> C: 24+24
        d2 = self.dec2(d2)              # Bx24x32x32

        # Block 3
        d3 = self.up3(d2)               # Bx24x64x64
        d3 = self.reduce2(d3)           # Bx16x64x64
        d3 = torch.cat([d3, x1], dim=1) # Bx32x64x64 --> C: 16+16
        d3 = self.dec3(d3)              # Bx16x64x64
        
        return self.head(d3)

# Quick Size Check
model = MicroSAM()
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")