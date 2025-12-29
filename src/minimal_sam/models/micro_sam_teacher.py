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
        reduced_c = max(1, in_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # BxCx1x1
            nn.Conv2d(in_channels, reduced_c, 1),  # Bx(reduced_c)x1x1
            nn.ReLU(inplace=True),                 # Bx(reduced_c)x1x1
            nn.Conv2d(reduced_c, in_channels, 1),  # BxCx1x1
            nn.Hardsigmoid()                       # BxCx1x1
        )

    def forward(self, x):
        return x * self.se(x)  # channel-wise multiplication (attention)


class InvertedResidual(nn.Module):
    """
    MobileNetV2/V3 style block: Expand -> Depthwise -> Project

    We use ReLU6 activations for better quantization compatibility.
    """
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, use_se=True):
        super().__init__()
        hidden_c = int(round(in_c * expand_ratio))
        self.use_res_connect = (stride == 1 and in_c == out_c)

        layers = []
        # 1. Expansion Phase (Pointwise 1x1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_c, hidden_c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_c))
            layers.append(nn.ReLU6(inplace=True))

        # 2. Depthwise Convolution
        layers.append(nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=stride, padding=1, groups=hidden_c, bias=False))
        layers.append(nn.BatchNorm2d(hidden_c))
        layers.append(nn.ReLU6(inplace=True))

        # 3. Squeeze and Excite (Optional but recommended)
        if use_se:
            layers.append(SqueezeExcite(hidden_c))

        # 4. Projection Phase (Pointwise 1x1)
        layers.append(nn.Conv2d(hidden_c, out_c, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_c))

        self.invresidual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.invresidual(x)
        else:
            return self.invresidual(x)


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, drop=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            # First Conv
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            # Second Conv for extra refinement
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.conv(x))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    Captures multi-scale context by applying parallel dilated convolutions with different rates.
    """
    def __init__(self, in_c, out_c, rates=(1, 2, 4, 6), drop=0.0):
        super().__init__()
        self.branches = nn.ModuleList()

        # Atrous (dilated) convolutions
        for rate in rates:
            if rate == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.SiLU(inplace=True)
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_c, out_c, kernel_size=3, padding=rate, dilation=rate, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.SiLU(inplace=True)
                    )
                )

        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

        # Project all layers
        # 5 branches: 1x1 + 3 dilated + GAP
        self.project = nn.Sequential(
            nn.Conv2d(out_c * (len(rates) + 1), out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        )

    def forward(self, x):
        res = []
        for branch in self.branches:
            res.append(branch(x))

        gap = self.global_avg_pool(x)
        gap = nn.functional.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(gap)

        x = torch.cat(res, dim=1)
        return self.project(x)


class Stem(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = out_c // 2
        # Channels: Input -> Mid -> Mid -> Output
        self.stem = nn.Sequential(
            # First Conv: 96x96 -> 48x48
            nn.Conv2d(in_c, mid_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True),
            # Second Conv
            nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True),
            # Third Conv
            nn.Conv2d(mid_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.stem(x)
            

class MicroSAMTeacher(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, width_mult=1.0):
        super().__init__()

        def c(ch): return max(8, int(ch * width_mult // 8) * 8)

        # Initial stem: 96x96 -> 48x48
        self.stem = Stem(in_channels, c(32))

        # Encoder Stages
        self.enc1 = nn.Sequential(
            InvertedResidual(c(32), c(32), stride=1, expand_ratio=4),
            InvertedResidual(c(32), c(32), stride=1, expand_ratio=4),
        ) # 48x48 -> 48x48

        self.enc2 = nn.Sequential(
            InvertedResidual(c(32), c(64), stride=2, expand_ratio=6),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=6),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=6),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=6)
        ) # 48x48 -> 24x24

        self.enc3 = nn.Sequential(
            InvertedResidual(c(64), c(128), stride=2, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
            InvertedResidual(c(128), c(128), stride=1, expand_ratio=6),
        ) # 24x24 -> 12x12

        self.enc4 = nn.Sequential(
            InvertedResidual(c(128), c(256), stride=2, expand_ratio=6),
            InvertedResidual(c(256), c(256), stride=1, expand_ratio=6),
            InvertedResidual(c(256), c(256), stride=1, expand_ratio=6),
            InvertedResidual(c(256), c(256), stride=1, expand_ratio=6),
        ) # 12x12 -> 6x6 (bottleneck)

        # Bottlenech ASPP
        self.aspp = ASPP(c(256), c(256), rates=(1, 2, 4, 6), drop=0.4)

        # Decoder Stages
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Decoder(c(256)+c(128), c(128), drop=0.05)
        self.dec2 = Decoder(c(128)+c(64),  c(64),  drop=0.05)
        self.dec3 = Decoder(c(64) +c(32),  c(32),  drop=0.05)

        # Refine back to 96x96
        self.refine = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c(32), c(32), kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(c(32)),
            nn.SiLU(inplace=True),
        )

        # Head
        self.head = nn.Conv2d(c(32), num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)               # (B,c(32),48,48)
        x1 = self.enc1(x1)              # (B,c(32),48,48)
        x2 = self.enc2(x1)              # (B,c(64),24,24)
        x3 = self.enc3(x2)              # (B,c(128),12,12)
        x4 = self.enc4(x3)              # (B,c(256),6,6)

        # Bottleneck
        b = self.aspp(x4)               # (B,c(256),6,6)

        # Decoder with Skip Connections
        d1 = self.up(b)                 # (B,c(256),12,12)
        d1 = torch.cat([d1, x3], dim=1) # (B,c(384),12,12) --> C: c(256) + c(128)
        d1 = self.dec1(d1)              # (B,c(128),12,12)

        d2 = self.up(d1)                # (B,c(128),24,24)
        d2 = torch.cat([d2, x2], dim=1) # (B,c(192),24,24) --> C: c(128) + c(64)
        d2 = self.dec2(d2)              # (B,c(64),24,24)

        d3 = self.up(d2)                # (B,c(64),48,48)
        d3 = torch.cat([d3, x1], dim=1) # (B,c(96),48,48)  --> C: c(64) + c(32)
        d3 = self.dec3(d3)              # (B,c(32),48,48)

        out = self.refine(d3)           # (B,c(32),96,96)

        return self.head(out)           # (B,num_classes,96,96)


if __name__ == "__main__":
    model = MicroSAMTeacher(in_channels=3, num_classes=1, width_mult=1.0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")  