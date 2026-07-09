# ================================================================
# Enhanced U-Net model for multi-modal BraTS 2021 segmentation
# ================================================================

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two convolutional layers with Batch Normalization and ReLU.
    Optional dropout is used in deeper layers.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EnhancedUNet(nn.Module):
    """
    Enhanced 2D U-Net for multi-modal brain tumor segmentation.

    Input:
        4-channel MRI tensor: T1, T1ce, T2, FLAIR

    Output:
        1-channel binary tumor mask
    """

    def __init__(self, in_channels=4, out_channels=1):
        super(EnhancedUNet, self).__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256, dropout=0.1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        self.bridge = ConvBlock(512, 1024, dropout=0.3)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512, dropout=0.2)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256, dropout=0.1)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bridge(p4)

        # Decoder
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        output = self.final_conv(d1)

        return output