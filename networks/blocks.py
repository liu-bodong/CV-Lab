import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DoubleConvBlock(nn.Module):
    """Double convolution block. 
    Input passes through the following layers:
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
    """Upsampling convolution block. 
    Input passes through the following layers:
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
    

class DepthWiseConvBlock(nn.Module):
    """Depth-wise convolution block. 
    Input passes through the following layers:
    3x3 Depth-wise Conv -> BN -> ReLU -> 1x1 Point-wise Conv -> BN -> ReLU

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.dw_conv(x)
    

class AsymmetricConv(nn.Module):
    """Asymmetric horizontal-vertical convolution block. 
    Input passes through the following layers:
    1x1 or 1x3 Conv -> BN -> ReLU

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mode (str): Asymmetry mode, either 'h' or 'v'.
    """
    def __init__(self, in_channels, out_channels, mode):
        super().__init__()
        self.asy_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3) if mode == 'h' else (3, 1), padding=(0, 1) if mode == 'h' else (1, 0), stride=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.asy_conv(x)))