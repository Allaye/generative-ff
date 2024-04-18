import torch
from generator_ff import *
from discriminators_ff import *


def test():
    x = torch.randn((1, 3, 256, 256))
    model = FFConvGenerator(in_channels=3, out_channels=64)
    preds = model(x)
    print(preds.shape)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset('data/facades/test/')
    dataloader = dataset.load_data(dataset, batch_size=32, shuffle=True)
    num_epochs = 20
    for epoch in range(num_epochs):
        for idx, (negative, positive) in enumerate(dataloader):
            pass

    test()
