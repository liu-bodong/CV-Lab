import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from thop import profile
# from thop import clever_format

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn    = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.bn(self.conv2(x)))
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv     = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn       = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)
    
class UNet(nn.Module):
    def __init__(self, img_channels, num_classes):
        super().__init__()
        self.conv_block1 = ConvBlock(img_channels, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        
        self.up_conv_block1 = UpConvBlock(1024, 512)
        self.conv_block6 = ConvBlock(1024, 512)
        
        self.up_conv_block2 = UpConvBlock(512, 256)
        self.conv_block7 = ConvBlock(512, 256)
        
        self.up_conv_block3 = UpConvBlock(256, 128)
        self.conv_block8 = ConvBlock(256, 128)
        
        self.up_conv_block4 = UpConvBlock(128, 64)
        self.conv_block9 = ConvBlock(128, 64)
        
        self.one_by_one = nn.Conv2d(64, num_classes, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        # Contracting path

        skip1 = self.conv_block1(x)

        skip2 = self.conv_block2(self.max_pool(skip1))

        skip3 = self.conv_block3(self.max_pool(skip2))

        skip4 = self.conv_block4(self.max_pool(skip3))

        x = self.conv_block5(self.max_pool(skip4))
        
        # Expansive path        
        x = self.up_conv_block1(x)
        x = self.conv_block6(torch.cat([x, skip4], dim=1))

        x = self.up_conv_block2(x)
        x = self.conv_block7(torch.cat([x, skip3], dim=1))

        x = self.up_conv_block3(x)
        x = self.conv_block8(torch.cat([x, skip2], dim=1))
        
        x = self.up_conv_block4(x)
        x = self.conv_block9(torch.cat([x, skip1], dim=1))
        
        x = self.one_by_one(x)
        return x
        

# if __name__ == "__main__":
#     with torch.no_grad():
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         input = torch.rand(1, 3, 256, 256).to(device)
#         model = UNet(3, 1).to(device)
#         flops, params = profile(model, (input,))

#         print("-" * 30)
#         print(f'Flops  = {clever_format(flops, format="%.5f")}')
#         print(f'Params = {clever_format(params, format="%.5f")}')
