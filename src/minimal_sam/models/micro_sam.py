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
    def __init__(self, in_channels, reduction=16):
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

        def decoder(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False), # Depthwise
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False), # Pointwise
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True),
            )

        # Initial stem: 96x96 -> 48x48 (reduce size early to save computations)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # Encoder Stages
        # Stage 1: 48x48 -> 48x48
        self.enc1 = InvertedResidual(16, 16, stride=1, expand_ratio=2)

        # Stage 2: 48x48 -> 24x24
        self.enc2 = nn.Sequential(
            InvertedResidual(16, 32, stride=2, expand_ratio=4),
            InvertedResidual(32, 32, stride=1, expand_ratio=4)
        )

        # Stage 3: 24x24 -> 12x12
        self.enc3 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=4),
            InvertedResidual(64, 64, stride=1, expand_ratio=4)
        )

        # Stage 4: 12x12 -> 6x6 (bottleneck)
        self.enc4 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=4),
            InvertedResidual(128, 128, stride=1, expand_ratio=4)
        )

        # Decoder Stages with Reduce and Skip Connections
        # Up 1: 6x6 -> 12x12
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce4 = nn.Conv2d(128, 64, 1, bias=False) # Reduce bottleneck channels
        self.dec1 = decoder(64+64, 64)

        # Up 2: 12x12 -> 24x24
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce3 = nn.Conv2d(64, 32, 1, bias=False)
        self.dec2 = decoder(32+32, 32)

        # Up 3: 24x24 -> 48x48
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce2 = nn.Conv2d(32, 16, 1, bias=False)
        self.dec3 = decoder(16+16, 16)

        # Up 4: 48x48 -> 96x96 (smooth final upsample)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),  # Depthwise smooth
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # Head
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)               # (B,16,48,48)
        x1 = self.enc1(x1)              # (B,16,48,48)
        x2 = self.enc2(x1)              # (B,32,24,24)
        x3 = self.enc3(x2)              # (B,64,12,12)
        x4 = self.enc4(x3)              # (B,128,6,6)
        
        # Decoder with Skip Connections
        d1 = self.up1(x4)               # (B,128,12,12)
        d1 = self.reduce4(d1)           # (B,64,12,12)
        d1 = torch.cat([d1, x3], dim=1) # (B,128,12,12) --> C: 64+64
        d1 = self.dec1(d1)              # (B,64,12,12)

        d2 = self.up2(d1)               # (B,64,24,24)
        d2 = self.reduce3(d2)           # (B,32,24,24)
        d2 = torch.cat([d2, x2], dim=1) # (B,64,24,24) --> C: 32+32
        d2 = self.dec2(d2)              # (B,32,24,24)

        d3 = self.up3(d2)               # (B,32,48,48)
        d3 = self.reduce2(d3)           # (B,16,48,48)
        d3 = torch.cat([d3, x1], dim=1) # (B,32,48,48) --> C: 16+16
        d3 = self.dec3(d3)              # (B,16,48,48)

        out = self.up4(d3)              # (B,16,96,96)

        return self.head(out)           # (B,num_classes,96,96)


if __name__ == "__main__":
    model = MicroSAM()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}") 