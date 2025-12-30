###################################################################################################
#
# Model definition for ai8x-training
# Image segmentation network
# Cyril Scherrer, 2025
#
###################################################################################################

"""
UNet network for MAX7800X
"""
import torch
from torch import nn

import ai8x

class MinimalSam(nn.Module):
    """
    Small size UNet model
    """
    def __init__(
            self,
            num_classes=1,
            num_channels=3,
            dimensions=(96, 96),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.enc1 = ai8x.FusedConv2dBNReLU(num_channels, 4, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv2dBNReLU(4, 8, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 32, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv3 = ai8x.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.dec3 = ai8x.FusedConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv2 = ai8x.ConvTranspose2d(32, 8, 3, stride=2, padding=1)
        self.dec2 = ai8x.FusedConv2dBNReLU(16, 8, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv1 = ai8x.ConvTranspose2d(8, 4, 3, stride=2, padding=1)
        self.dec1 = ai8x.FusedConv2dBNReLU(8, 16, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv = ai8x.FusedConv2dBN(16, num_classes, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        bottleneck = self.bneck(enc3)

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return self.conv(dec1)
    
def minimalsam(pretrained=False, **kwargs):
    """
    Constructs a small unet (unet_v3) model.
    """
    assert not pretrained
    return MinimalSam(**kwargs)
    
models = [
    {
        'name': 'minimalsam',
        'min_input': 1,
        'dim': 2,
    },
]