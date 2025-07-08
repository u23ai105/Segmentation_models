import torch
import torch.nn as nn

from unet_parts import DoubleConv,DownSample,UpSample

class UNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.down_convolution_1=DownSample(in_channels,64)
        self.down_convolution_2=DownSample(1,128)
        self.down_convolution_3 = DownSample(in_channels, 256)
        self.down_convolution_4 = DownSample(in_channels, 512)

        self.bootle_neck=DoubleConv(512,1024)

        self.up_convolution_1=UpSample(1024,512)
        self.up_convolution_2=UpSample(512,256)
        self.up_convolution_3=UpSample(256,128)
        self.up_convolution_4=UpSample(128,64)

        self.out=nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=1)

    def forward(self,x):
        down1,p1=self.down_convolution_1(x)
        down2,p2=self.down_convolution_2(p1)
        down3,p3=self.down_convolution_2(p2)
        down4,p4=self.down_convolution_2(p3)

        b=self.bottle_neck(down4)

        up1=self.up_convolution_1(b,p4)
        up2=self.up_convolution_2(up1,p3)
        up3=self.up_convolution_3(up2,p2)
        up4=self.up_convolution_4(up3,p1)

        out=self.out(up4)
        return out
