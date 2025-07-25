import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConvBlock(nn.Module):
    """Convolution block in U-Net. Input passes through the following layers:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

   
class UpConvBlock(nn.Module):
    """Upsampling convolution block in U-Net. Input passes through the following layers:
    Upsample -> Conv2d -> BatchNorm2d -> ReLU

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up_conv(x)
    

