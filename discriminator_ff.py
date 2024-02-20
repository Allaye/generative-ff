import torch
import torch.nn as nn
from layers_ff import FFLinearLayer
from utils import one_hot_encode


class BaseDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass


class FFConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


class FFDenseDiscriminator(nn.Module):
    """
    The feed forward dense discriminator.
    """

    def __init__(self, dimension, num_epoch, output_dim=2):
        super().__init__()
        self.layers = []
        self.num_epoch = num_epoch
        self.output_dim = output_dim
        for dim in range(len(dimension) - 1):
            self.layers += [FFLinearLayer(dimension[dim], dimension[dim + 1])]
            # self.layer.append(nn.Linear(dimension[dim], dimension[dim + 1]))

    def predict(self, x):
        goodness_score_per_label = []
        for label in range(self.output_dim):
            # perform one hot encoding#
            encoded = one_hot_encode(x, label)
            goodness = []
            for layer in self.layers:
                encoded = layer(encoded)
                goodness += [encoded.pow(2).mean(1)]
            goodness_score_per_label += [sum(goodness).unsqueeze(1)]
        goodness_score_per_label = torch.cat(goodness_score_per_label, 1)
        return goodness_score_per_label.argmax(1)

    def train(self, x_positive, x_negative):
        for i, layer in enumerate(self.layers):
            # x_positive = layer(x_positive)
            # x_negative = layer(x_negative)
            print('training layer', i, '...')
            x_positive, x_negative = layer.forward_forward(x_positive, x_negative)


# from torchvision.datasets import MNIST
# from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
# from torch.utils.data import DataLoader
#
#
# def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
#     transform = Compose([
#         ToTensor(),
#         Normalize((0.1307,), (0.3081,)),
#         Lambda(lambda x: torch.flatten(x))])
#
#     train_loader = DataLoader(
#         MNIST('./data/', train=True,
#               download=True,
#               transform=transform),
#         batch_size=train_batch_size, shuffle=True)
#
#     test_loader = DataLoader(
#         MNIST('./data/', train=False,
#               download=True,
#               transform=transform),
#         batch_size=test_batch_size, shuffle=False)
#
#     return train_loader, test_loader
#
#
# # Instantiate the FFDenseDiscriminator
# dense_discriminator = FFDenseDiscriminator([784, 500, 500], 100)
#
# # # Generate random input data (adjusted for the input feature as needed)
# train_loader, test_loader = MNIST_loaders()
# x, y = next(iter(train_loader))
# x, y = x, y
# x_pos = one_hot_encode(x, y)
# rnd = torch.randperm(x.size(0))
# x_neg = one_hot_encode(x, y[rnd])
#
# dense_discriminator.train(x_pos, x_neg)
#
# print('train error:', 1.0 - dense_discriminator.predict(x).eq(y).float().mean().item())
#
# x_te, y_te = next(iter(test_loader))
# x_te, y_te = x_te, y_te
#
# print('test error:', 1.0 - dense_discriminator.predict(x_te).eq(y_te).float().mean().item())