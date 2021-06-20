import torch
from torch import mode, nn, reshape
from torch.nn.modules.conv import Conv2d
from torchvision import models
import torch.nn.functional as F
import numpy as np


class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.elu(x)
        return x


class UpConv(nn.Module):
    """ Up-Sample Section """

    def __init__(self, in_channels, out_channels, kernel_size, scale):
        super().__init__()
        self.scale = scale
        self.conv = ConvBN(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        # Up-Sample used imterpolate op
        x = F.interpolate(inputs, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class ResUNet(nn.Module):
    """ Partical ResNet-x """
    _DEFAULT_NAMES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    def __init__(self, name, coarse_dim, fine_dim):
        super().__init__()
        if name not in self._DEFAULT_NAMES:
            raise ValueError('Incorrect backbone name')
        if name in self._DEFAULT_NAMES:
            channels = [64, 128, 256, 512]  # 0, 1, 2, 3
        else:
            channels = [256, 512, 1024, 2048]
        resnet = getattr(models, name)(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        # Encoder
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder
        self.upconv3 = UpConv(channels[2], 512, 3, 2)
        self.iconv3 = ConvBN(channels[1] + 512, 512, 3, 1)
        self.upconv2 = UpConv(512, 256, 3, 2)
        self.iconv2 = ConvBN(channels[0] + 256, 256, 3, 1)
        # coarse conv
        self.conv_coarse = ConvBN(channels[2], coarse_dim, 1, 1)
        # fine conv
        self.conv_fine = ConvBN(256, fine_dim, 1, 1)

    def skipconnect(self, x1, x2):
        diff_x = x2.size()[2] - x1.size()[2]  # difference of h
        diff_y = x1.size()[3] - x1.size()[3]  # difference of w
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([x2, x1], dim=1)  # dim
        return x

    def forward(self, inputs):
        x = self.firstrelu(self.firstbn(self.firstconv(inputs)))  # 64-d [1, 512, 120, 160]

        x1 = self.layer1(x)  # 64-d [1, 64, 240, 320]
        x2 = self.layer2(x1)  # 128-d [1, 128, 120, 160]
        x3 = self.layer3(x2)  # 256-d [1, 256, 60, 80]

        coarse_feature = self.conv_coarse(x3)  # coarse_dim-d [1, 1024, 60, 80]

        x = self.upconv3(x3)  # 512-d [1, 512, 120, 160]
        x = self.skipconnect(x2, x)  # 640-d [1, 640, 120, 160]
        x = self.iconv3(x)  # 512-d [1, 512, 120, 160]
        x = self.upconv2(x)  # 256-d [1, 256, 240, 320]
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)  # 256-d [1, 256, 240, 320]

        fine_feature = self.conv_fine(x)  # fine_dim-d [1, 1, 240, 320]

        return [coarse_feature, fine_feature]


