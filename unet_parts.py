import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_op=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=DoubleConv(in_channels,out_channels)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        down=self.conv(x)
        print(down.shape)
        p=self.pool(down)
        print(p.shape)
        return down,p

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv=DoubleConv(in_channels,out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        print(f"After upsample: x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        print(f"After padding: x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)