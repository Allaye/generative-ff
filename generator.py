"""
Code for research work generative forward-forward neural networks
Date 11/03/2024.
Author: Kolade Gideon *Allaye*
"""

import torch
from torch import nn
from layers_ff import FFConvLayer, FFConvTransLayer


class FFBaseGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, down=True,
                 act="relu", drop=False):
        super(FFBaseGeneratorBlock, self).__init__()
        self.conv = nn.Sequential(
            FFConvLayer(in_channels, out_channels, kernel_size, stride, padding, bias=bias, act=act, drop=drop)
            if down
            else FFConvTransLayer(in_channels, out_channels, kernel_size, stride, padding, bias=bias, act=act),
        )

    def forward(self, x):
        return self.conv(x)


class FFConvGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFConvGenerator, self).__init__()
        self.initial_down = nn.Sequential(
            FFConvLayer(in_channels, out_channels, 4, 2, 1, padding_mode="reflect", init=True),
        )
        self.down1 = FFBaseGeneratorBlock(out_channels, out_channels * 2, down=True, act="leaky")
        self.down2 = FFBaseGeneratorBlock(out_channels * 2, out_channels * 4, down=True, act="leaky")
        self.down3 = FFBaseGeneratorBlock(out_channels * 4, out_channels * 8, down=True, act="leaky")
        self.down4 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=True, act="leaky")
        self.down5 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=True, act="leaky")
        self.down6 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=True, act="leaky")
        self.bottleneck = nn.Sequential(
            FFConvLayer(out_channels * 8, out_channels * 8, 4, 2, 1, act="relu", padding_mode="reflect")
        )
        self.up1 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=False, act="relu", drop=True)
        self.up2 = FFBaseGeneratorBlock(out_channels * 8 * 2, out_channels * 8, down=False, act="relu", drop=True)
        self.up3 = FFBaseGeneratorBlock(out_channels * 8 * 2, out_channels * 8, down=False, act="relu", drop=True)
        self.up4 = FFBaseGeneratorBlock(out_channels * 8 * 2, out_channels * 8, down=False, act="relu")
        self.up5 = FFBaseGeneratorBlock(out_channels * 8 * 2, out_channels * 4, down=False, act="relu")
        self.up6 = FFBaseGeneratorBlock(out_channels * 4 * 2, out_channels * 2, down=False, act="relu")
        self.up7 = FFBaseGeneratorBlock(out_channels * 2 * 2, out_channels, down=False, act="relu")
        self.final_up = nn.Sequential(
            FFConvTransLayer(out_channels * 2, out_channels, 4, 2, 1, ),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        print('d4', d4.shape)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        print('d7', d7.shape)
        bottleneck = self.bottleneck(d7)
        print('bottleneck', bottleneck.shape)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        print('up7', up7.shape)
        print('d1', d1.shape)
        return self.final_up(torch.cat([up7, d1], 1))


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    # pass
    # # print('hello world')
    # model = FFBaseGeneratorBlock(1, 64, 4, 2, 1)
    # data = torch.rand(32, 1, 28, 28)
    # print(model)
    #
    # print(model(data).shape)
    x = torch.randn(32, 3, 256, 259)
    model = TestModel()
    print(model)
    preds = model(x)
    print(preds.shape)

    # model = FFConvGenerator(in_channels=3, out_channels=64)
    # print(model)
    # preds = model(x)
    # print(preds.shape)

    #     print(data.shape, label.shape)
