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
    Large size UNet model. This model also enables the use of folded data.
    """
    def __init__(
            self,
            num_classes=2,
            num_channels=3,
            dimensions=(88, 88),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.enc1 = ai8x.FusedConv2dBNReLU(num_channels, 8, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv2dBNReLU(8, 28, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv2dBNReLU(28, 56, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck0 = ai8x.FusedMaxPoolConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)
        self.bneck1 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)
        self.bneck2 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)
        self.bneck3 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)
        self.bneck4 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)
        self.bneck5 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)
        self.bneck6 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)        
        self.bneck7 = ai8x.FusedConv2dBNReLU(56, 56, 3, stride=1, padding=1,
                                                    bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv3 = ai8x.ConvTranspose2d(56, 56, 3, stride=2, padding=1)
        self.dec3 = ai8x.FusedConv2dBNReLU(112, 56, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv2 = ai8x.ConvTranspose2d(56, 28, 3, stride=2, padding=1)
        self.dec2 = ai8x.FusedConv2dBNReLU(56, 28, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv1 = ai8x.ConvTranspose2d(28, 8, 3, stride=2, padding=1)
        self.dec1 = ai8x.FusedConv2dBNReLU(16, 64, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.dec0 = ai8x.FusedConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv = ai8x.FusedConv2dBN(32, num_classes, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN

        enc1 = self.enc1(x)                    # 8x(dim1)x(dim2)
        enc2 = self.enc2(enc1)                 # 28x(dim1/2)x(dim2/2)
        enc3 = self.enc3(enc2)                 # 56x(dim1/4)x(dim2/4)

        bneck0 = self.bneck0(enc3)          # 56x(dim1/8)x(dim2/8)
        bneck1 = self.bneck1(bneck0)
        bneck2 = self.bneck1(bneck1)
        bneck3 = self.bneck1(bneck2)
        bneck4 = self.bneck1(bneck3)
        bneck5 = self.bneck1(bneck4)
        bneck6 = self.bneck1(bneck5)
        bneck7 = self.bneck1(bneck6)

        dec3 = self.upconv3(bneck7)        # 56x(dim1/4)x(dim2/4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 112x(dim1/4)x(dim2/4)
        dec3 = self.dec3(dec3)                 # 56x(dim1/4)x(dim2/4)
        dec2 = self.upconv2(dec3)              # 28x(dim1/2)x(dim2/2)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 56x(dim1/2)x(dim2/2)
        dec2 = self.dec2(dec2)                 # 28x(dim1/2)x(dim2/2)
        dec1 = self.upconv1(dec2)              # 8x(dim1)x(dim2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 16x(dim1)x(dim2)
        dec1 = self.dec1(dec1)                 # 48x(dim1)x(dim2)

        dec0 = self.dec0(dec1)                 # 32x(dim1)x(dim2)
        dec0 = self.conv(dec0)                 # num_final_channelsx(dim1)x(dim2)

        return dec0
    
def minimalsam(pretrained=False, **kwargs):
    """
    Constructs a unet model for image segmentation.
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