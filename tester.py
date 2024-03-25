import torch
from generator import *


def test():
    x = torch.randn((1, 3, 256, 256))
    model = FFConvGenerator(in_channels=3, out_channels=64)
    preds = model(x)
    print(preds.shape)


if __name__ == '__main__':
    test()
