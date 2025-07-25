import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from networks.blocks import ConvBlock, UpConvBlock

class AttentionGate(nn.Module):
    """Attention Gate proposed in Attention U-Net paper. 
    
    x' = W_x(x) -> ReLU(x' + g') -> Psi() -> sigmoid() -> upsample -> a
    returns a * x

    Args:
        F_g (int): Number of input channels for gating signal.
        F_x (int): Number of input channels for the feature map.
        F_int (int): Number of intermediate channels used in the attention mechanism.
    """
    def __init__(self, F_g, F_x, F_int, mode='bilinear'):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, padding=0, stride=2),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(1)
        )
        
        # upsample back to F_x channels
        self.resampler = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        
    def forward(self, x, g):
        x1 = self.W_x(x)
        g = self.W_g(g)
        psi = self.psi(torch.relu(x1 + g))
        psi = torch.sigmoid(psi)
        a =  self.resampler(psi)
        return a * x

class AttnUpConvBlock(nn.Module):
    """Upsampling convolution block with attention for U-Net.

    Args:
        ch_last (int): Number of channels from the previous (decoder) layer.
        ch_skip (int): Number of channels from the encoder skip connection.
        ch_inter (int): Number of intermediate channels for the attention gate.
        ch_out (int): Number of output channels after convolution.
    """
    def __init__(self, ch_last, ch_skip, ch_inter, ch_out):
        super().__init__()
        self.attention_gate = AttentionGate(ch_last, ch_skip, ch_inter)
        self.upconv = UpConvBlock(ch_last, ch_skip)
        self.conv = ConvBlock(ch_last, ch_out)

    def forward(self, x, g):        
        a = self.attention_gate(x, g)
        d = torch.cat((self.upconv(g), a), dim=1)
        return self.conv(d)
        

class AttnUNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels = [64, 128, 256, 512]):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_conv = ConvBlock(in_channels, channels[0])
        
        self.encoder_path = nn.ModuleList(
            [ConvBlock(channels[i], channels[i+1]) 
             for i in range(0, len(channels) - 1)])
        
        self.decoder_path = nn.ModuleList(
            [AttnUpConvBlock(ch_last=channels[i], 
                             ch_skip=channels[i-1], 
                             ch_inter=channels[i-1], 
                             ch_out=channels[i-1]) 
            for i in range(len(channels) -1, 0, -1)])
        
        self.conv_1x1 = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        # Encoder
        
        x = self.in_conv(x)
        
        skips = []
        skips.append(x)
        
        for idx, blk in enumerate(self.encoder_path):
            x = blk(self.maxpool(x))
            # print(f"block {idx+1}", x.shape)
            if idx < len(self.encoder_path) - 1:
                skips.append(x)
                        
        g = x
            
        # Decoder
        # skips = skips[:-1]
        skips.reverse()
        
        # print("skips", [s.shape for s in skips])
        
        for idx, blk in enumerate(self.decoder_path):
            x = skips[idx]
            g = blk(x, g)
        
        return self.conv_1x1(g)
        
        