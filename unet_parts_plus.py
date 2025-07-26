import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels,dropout=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels ,dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out

class UpSample(nn.Module):
    def __init__(self, up_in_ch, conv_in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_ch, up_in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(conv_in_ch, out_ch, dropout=dropout)

    def forward(self, x1, *skips):
        x1 = self.up(x1)

        if skips:
            target_skip = skips[0]
            diffY = target_skip.size(2) - x1.size(2)
            diffX = target_skip.size(3) - x1.size(3)

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([*skips, x1], dim=1)
        return self.conv(x)