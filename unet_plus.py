import torch
import torch.nn as nn
from unet_parts_plus import DownSample, UpSample, DoubleConv

class UNetPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512,1024]):
        super().__init__()
        f = features

        # Encoder Path
        self.down0 = DownSample(in_channels, f[0])
        self.down1 = DownSample(f[0], f[1])
        self.down2 = DownSample(f[1], f[2])
        self.down3 = DownSample(f[2], f[3])

        self.bottleneck = DoubleConv(f[3], f[3] * 2)
        # Decoder path (nested skip connections)
        # The node X_i,j is the output of the j-th conv layer on the i-th level
        # conv_in_ch = (channels from horizontal skips) + (channels from upsampled tensor)
        # up_in_ch = channels of the tensor being upsampled

        # Level j=1
        self.up01 = UpSample(up_in_ch=f[1], conv_in_ch=f[0] + f[1]//2, out_ch=f[0])
        self.up11 = UpSample(up_in_ch=f[2], conv_in_ch=f[1] + f[2]//2, out_ch=f[1])
        self.up21 = UpSample(up_in_ch=f[3], conv_in_ch=f[2] + f[3]//2, out_ch=f[2])
        self.up31 = UpSample(up_in_ch=f[4], conv_in_ch=(f[3] + f[4]//2), out_ch=f[3])

        # Level j=2
        self.up02 = UpSample(up_in_ch=f[1], conv_in_ch=f[0]*2 + f[1]//2, out_ch=f[0])
        self.up12 = UpSample(up_in_ch=f[2], conv_in_ch=f[1]*2 + f[2]//2, out_ch=f[1])
        self.up22 = UpSample(up_in_ch=f[3], conv_in_ch=f[2]*2 + f[3]//2, out_ch=f[2])

        # Level j=3
        self.up03 = UpSample(up_in_ch=f[1], conv_in_ch=f[0]*3 + f[1]//2, out_ch=f[0])
        self.up13 = UpSample(up_in_ch=f[2], conv_in_ch=f[1]*3 + f[2]//2, out_ch=f[1])

        # Level j=4
        self.up04 = UpSample(up_in_ch=f[1], conv_in_ch=f[0]*4 + f[1]//2, out_ch=f[0])

        self.final_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)
        self.out01 = nn.Conv2d(f[0], out_channels, 1)
        self.out02 = nn.Conv2d(f[0], out_channels, 1)
        self.out03 = nn.Conv2d(f[0], out_channels, 1)
        self.out04 = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        # Encoder
        x00, p0 = self.down0(x)
        x10, p1 = self.down1(p0)
        x20, p2 = self.down2(p1)
        x30, p3 = self.down3(p2)
        x40 = self.bottleneck(p3)

        # Decoder path
        # Level j=1
        x01 = self.up01(x10, x00)
        x11 = self.up11(x20, x10)
        x21 = self.up21(x30, x20)
        x31 = self.up31(x40, x30)

        # Level j=2
        x02 = self.up02(x11, x00, x01)
        x12 = self.up12(x21, x10, x11)
        x22 = self.up22(x31, x20, x21)

        # Level j=3
        x03 = self.up03(x12, x00, x01, x02)
        x13 = self.up13(x22, x10, x11, x12)

        # Level j=4
        x04 = self.up04(x13, x00, x01, x02, x03)

        out01 = self.out01(x01)
        out02 = self.out02(x02)
        out03 = self.out03(x03)
        out04 = self.out04(x04)

        return [out01, out02, out03, out04]