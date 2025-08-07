import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from networks.blocks import DepthWiseConvBlock

class MobileNetV1(nn.Module):
    def __init__(self, in_channels, n_classes, channels=[32,64,128,256,512,1024]):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(
            DepthWiseConvBlock(in_channels, 32, 2),
            DepthWiseConvBlock(32, 64, 1),
            DepthWiseConvBlock(64, 128, 2),
            DepthWiseConvBlock(128, 128, 1),
            DepthWiseConvBlock(128, 256, 2),
            DepthWiseConvBlock(256, 256, 1),
            DepthWiseConvBlock(256, 512, 2),
            DepthWiseConvBlock(512, 512, 1),
            DepthWiseConvBlock(512, 512, 1),
            DepthWiseConvBlock(512, 512, 1),
            DepthWiseConvBlock(512, 512, 1),
            DepthWiseConvBlock(512, 512, 1),
            DepthWiseConvBlock(512, 1024, 2),
            DepthWiseConvBlock(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
