from torch import nn


class PicoSAM(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def depthwise_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder_stage1 = depthwise_conv(in_channels, 48)
        self.down1 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_stage2 = depthwise_conv(48, 96)
        self.down2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_stage3 = depthwise_conv(96, 160)
        self.down3 = nn.Conv2d(160, 160, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_stage4 = depthwise_conv(160, 256)
        self.down4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)

        self.bottleneck = depthwise_conv(256, 320)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(320, 192))
        self.skip_conn4 = nn.Conv2d(256, 192, kernel_size=1)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(192, 128))
        self.skip_conn3 = nn.Conv2d(160, 128, kernel_size=1)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(128, 80))
        self.skip_conn2 = nn.Conv2d(96, 80, kernel_size=1)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(80, 40))
        self.skip_conn1 = nn.Conv2d(48, 40, kernel_size=1)

        self.output_head = nn.Conv2d(40, 1, kernel_size=1)

    def forward(self, x):
        feat1 = self.encoder_stage1(x)
        feat2 = self.encoder_stage2(self.down1(feat1))
        feat3 = self.encoder_stage3(self.down2(feat2))
        feat4 = self.encoder_stage4(self.down3(feat3))
        bottleneck_out = self.bottleneck(self.down4(feat4))

        upsample1 = self.up1(bottleneck_out) + self.skip_conn4(feat4)
        upsample2 = self.up2(upsample1) + self.skip_conn3(feat3)
        upsample3 = self.up3(upsample2) + self.skip_conn2(feat2)
        upsample4 = self.up4(upsample3) + self.skip_conn1(feat1)

        return self.output_head(upsample4)
    

if __name__ == "__main__":
    model = PicoSAM()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params / 1e3:.1f}k")