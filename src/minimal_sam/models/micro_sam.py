import torch
import torch.nn as nn

from torchvision.ops import SqueezeExcitation


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, use_se=True):
        super(InvertedResidual, self).__init__()
        hidden_c = int(round(in_c * expand_ratio))
        self.use_res_connect = (stride == 1 and in_c == out_c)

        layers = []
        # 1. Expansion Phase (Pointwise 1x1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_c, hidden_c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_c))
            layers.append(nn.Hardswish(inplace=True))

        # 2. Depthwise Convolution
        layers.append(nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=stride, padding=1, groups=hidden_c, bias=False))
        layers.append(nn.BatchNorm2d(hidden_c))
        layers.append(nn.Hardswish(inplace=True))

        # 3. Squeeze and Excite (Optional: reduction ratio 16)
        if use_se:
            layers.append(SqueezeExcitation(hidden_c, hidden_c // 16, nn.ReLU6, nn.Hardsigmoid))

        # 4. Projection Phase (Pointwise 1x1) with Linear Activation!
        layers.append(nn.Conv2d(hidden_c, out_c, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.DWSconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=False), # Depthwise
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )
    
    def forward(self, x):
        return self.DWSconv(x)


class BilinearUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(BilinearUpsample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x):
        return self.conv(self.up(x))


class MicroSAM(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Initial stem: 96x96 -> 48x48 (reduce size early to save computations)
        self.stem = nn.Sequential(
            # First conv: reduce resolution
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            # Second conv: feature extraction
            DepthWiseSeparableConv2d(16, 16)
        )

        # Encoder Stages
        self.enc1 = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=2)
        ) # 48x48 -> 48x48

        self.enc2 = nn.Sequential(
            InvertedResidual(16, 32, stride=2, expand_ratio=4),
            InvertedResidual(32, 32, stride=1, expand_ratio=4)
        ) # 48x48 -> 24x24

        self.enc3 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=4),
            InvertedResidual(64, 64, stride=1, expand_ratio=4)
        ) # 24x24 -> 12x12

        self.enc4 = nn.Sequential(
            InvertedResidual(64, 128, stride=2, expand_ratio=4),
            InvertedResidual(128, 128, stride=1, expand_ratio=4)
        ) # 12x12 -> 6x6 (bottleneck)

        # Decoder Stages with Reduce and Skip Connections
        # Up 1: 6x6 -> 12x12
        self.up1 = BilinearUpsample(128, 64, scale_factor=2)
        self.dec1 = DepthWiseSeparableConv2d(64+64, 64)  

        # Up 2: 12x12 -> 24x24
        self.up2 = BilinearUpsample(64, 32, scale_factor=2)
        self.dec2 = DepthWiseSeparableConv2d(32+32, 32)
        
        # Up 3: 24x24 -> 48x48
        self.up3 = BilinearUpsample(32, 16, scale_factor=2)
        self.dec3 = DepthWiseSeparableConv2d(16+16, 16)

        # Up 4: 48x48 -> 96x96 (smooth final upsample)
        self.up4 = BilinearUpsample(16, 16, scale_factor=2)

        # Head
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)                           # (B,16,48,48)
        x1 = self.enc1(x1)                          # (B,16,48,48)
        x2 = self.enc2(x1)                          # (B,32,24,24)
        x3 = self.enc3(x2)                          # (B,64,12,12)
        x4 = self.enc4(x3)                          # (B,128,6,6)
        
        # Decoder with Skip Connections
        d1 = self.up1(x4)                           # (B,64,12,12)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))  # (B,64,12,12) --> C: 64+64

        d2 = self.up2(d1)                           # (B,32,24,24)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))  # (B,32,24,24) --> C: 32+32

        d3 = self.up3(d2)                           # (B,16,48,48)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))  # (B,16,48,48) --> C: 16+16

        out = self.up4(d3)                          # (B,16,96,96)

        return self.head(out)                       # (B,num_classes,96,96)


if __name__ == "__main__":
    model = MicroSAM()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}") 