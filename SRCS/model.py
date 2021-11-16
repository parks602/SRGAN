import math
import torch
from torch import nn
import sys

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, padding=4, padding_mode = 'replicate'),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(32)
        self.block3 = ResidualBlock(32)
        self.block4 = ResidualBlock(32)
        self.block6 = ResidualBlock(32)
        self.block7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode = 'replicate',bias=False),
            nn.BatchNorm2d(32)
        )
        block8 = [UpsampleBLock(32, 10)]
        block8.append(nn.Conv2d(32, 16, kernel_size=6, padding=0))
        self.block8 = nn.Sequential(*block8)

        self.landsea = nn.Sequential(
            nn.Conv2d(1,16, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
            nn.LeakyReLU()
        )
        self.hight  = nn.Sequential(
            nn.Conv2d(1,16, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
            nn.LeakyReLU()
        )

        self.Last  = nn.Sequential(
           nn.Conv2d(48, 48, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
           nn.LeakyReLU(),
           nn.Conv2d(48, 16, kernel_size = 3, padding = 1, padding_mode = 'replicate'),
           nn.LeakyReLU(),
           nn.Conv2d(16, 1, kernel_size = 3, padding = 1, padding_mode = 'replicate')
           )
    def forward(self, x, LS, HI):
        block1  = self.block1(x)
        block2  = self.block2(block1)
        block3  = self.block3(block2)
        block4  = self.block4(block3)
        block5  = self.block6(block4)
        block6  = self.block7(block5)
        block7  = self.block8(block1 + block6)
        LS      = self.landsea(LS)
        HI      = self.hight(HI)
        e = torch.cat((block7, LS, HI), 1)
        temp    = self.Last(e)
        return  temp

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, padding_mode = 'replicate', bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, padding_mode = 'replicate',bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode = 'replicate', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size, 1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode = 'replicate',bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode = 'replicate',bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.upsample = nn.Upsample(scale_factor = up_scale, mode = 'bilinear', align_corners = True)

    def forward(self, x):
        x = self.upsample(x)
        return x

