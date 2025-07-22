import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
    
from Models.blocks import ConvBlock, UpConvBlock

class ConcatUpConvBlock(nn.Module):
    def __init__(self, ch_last, ch_skip, ch_out):
        super().__init__()
        self.upconv = UpConvBlock(ch_last, ch_skip)
        self.conv = ConvBlock(ch_last, ch_out)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels = [64, 128, 256, 512]):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.in_conv = ConvBlock(in_channels, channels[0])
        
        self.encoder = nn.ModuleList(
            [ConvBlock(channels[i], channels[i+1]) 
             for i in range(0, len(channels) - 1)])
        
        self.decoder = nn.ModuleList(
            [ConcatUpConvBlock(ch_last=channels[i], 
                               ch_skip=channels[i-1],
                               ch_out=channels[i-1])
             for i in range(len(channels) - 1, 0, -1)])
        
        self.conv_1x1 = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        # Encoder
        
        x = self.in_conv(x)
        
        skips = []
        skips.append(x)
        
        for idx, blk in enumerate(self.encoder):
            x = blk(self.maxpool(x))
            if idx < len(self.encoder) - 1:
                skips.append(x)
        
        
        # Decoder
        
        skips.reverse()
        
        for idx, blk in enumerate(self.decoder):
            s = skips[idx]
            x = blk(x, s)
        
        x = self.conv_1x1(x)
        return x
        


