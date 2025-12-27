import torch
import torch.nn as nn

class MinimalSAM(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def FusedConv2dBNReLU(in_c, out_c, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=1, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.05, affine=False),
                nn.ReLU(inplace=True),
            )
        
        def FusedMaxPoolConv2dBNReLU(in_c, out_c, kernel_size, stride, padding):
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Conv2d(in_c, out_c, kernel_size, padding, groups=1, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.05, affine=False),
                nn.ReLU(inplace=True),
            )
        
        self.enc1 = FusedConv2dBNReLU(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.enc2 = FusedMaxPoolConv2dBNReLU(4, 8, kernel_size=3, stride=1, padding=1)
        self.enc3 = FusedMaxPoolConv2dBNReLU(8, 32, kernel_size=3, stride=1, padding=1)

        self.bneck = FusedMaxPoolConv2dBNReLU(32, 64, kernel_size=3, stride=1, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.dec3 = FusedConv2dBNReLU(64, 60, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose2d(60, 8, kernel_size=3, stride=2, padding=1)
        self.dec2 = FusedConv2dBNReLU(16, 48, kernel_size=3, stride=1, padding=1)

        self.upconv1 = nn.ConvTranspose2d(48, 4, kernel_size=3, stride=2, padding=1)
        self.dec1 = FusedConv2dBNReLU(8, 64, kernel_size=3, stride=1, padding=1)

        self.dec0 = FusedConv2dBNReLU(64, 32, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.05, affine=False)
        )
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        bneck = self.bneck(enc3)

        dec3 = self.upconv3(bneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        dec0 = self.dec0(dec1)
        return self.conv(dec0)