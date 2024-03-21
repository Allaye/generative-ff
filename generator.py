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
            FFConvLayer(in_channels, out_channels, 4, 2, 1, padding_mode="reflect"),
        )
        self.down1 = FFBaseGeneratorBlock(out_channels, out_channels * 2, down=True, act="leaky")
        self.down2 = FFBaseGeneratorBlock(out_channels * 2, out_channels * 4, down=True, act="leaky")
        self.down3 = FFBaseGeneratorBlock(out_channels * 4, out_channels * 8, down=True, act="leaky")
        self.down4 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=True, act="leaky")
        self.down5 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=True, act="leaky")
        self.down6 = FFBaseGeneratorBlock(out_channels * 8, out_channels * 8, down=True, act="leaky")
        self.bottleneck = nn.Sequential(
            FFConvLayer(in_channels, out_channels, 4, 2, 1, act="relu", padding_mode="reflect")
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


if __name__ == '__main__':
    pass
    # print('hello world')
    model = FFBaseGeneratorBlock(1, 64, 4, 2, 1)
    data = torch.rand(32, 1, 28, 28)
    print(model)

    print(model(data).shape)

    #     print(data.shape, label.shape)
