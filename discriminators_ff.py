"""
Code for research work generative forward-forward neural networks "
Date 20/02/2024.
Author: Kolade-Gideon *Allaye*
"""
import os
import torch
import torch.nn as nn
from layers_ff import FFLinearLayer, FFConvLayer
from utils import one_hot_encode_label_on_image, overlay_y_on_x, visualize_sample
from dataloader import CustomDisDataset
from torch.utils.data import Sampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class BaseDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass


class FFConvDiscriminator(nn.Module):
    def __init__(self, dimension, output_dim=10, kernel_size=3):
        super().__init__()
        self.layers = []
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        for dim in range(len(dimension) - 1):
            self.layers.append(FFConvLayer(dimension[dim], dimension[dim + 1], self.kernel_size, padding=self.padding))
        print(self.layers)

    def predict(self, x):
        goodness_score_per_label = []
        for label in range(self.output_dim):
            # perform one hot encoding#
            # x = torch.reshape(x, (x.shape[0], -1))
            # print('label:', label, x.shape)
            x = torch.reshape(x, (x.shape[0], -1))
            encoded = overlay_y_on_x(x, label)
            encoded = torch.reshape(encoded, (encoded.shape[0], 1, 28, 28))
            # print('encoded here :', encoded.shape)
            goodness = []
            for idx, layer in enumerate(self.layers):
                encoded = layer(encoded)
                shape = encoded.shape
                # print('encoded:', encoded.shape)
                encoded = torch.reshape(encoded, (encoded.shape[0], -1))
                # print('encoded:', encoded.shape)
                goodness += [encoded.pow(2).mean(1)]
                # goodness += [torch.reshape(encoded, (encoded.shape[0], -1)).pow(2).mean(1)]
                # print('goodness:', len(goodness), goodness[idx].shape)
                # print('sum goodness:', sum(goodness))
                encoded = encoded.reshape(shape)
            goodness_score_per_label += [sum(goodness).unsqueeze(1)]
            # print('goodness_score_per_label:', len(goodness_score_per_label))
        goodness_score_per_label = torch.cat(goodness_score_per_label, 1)
        return goodness_score_per_label.argmax(1)

    def train(self, x_positive, x_negative):
        for i, layer in enumerate(self.layers):
            # x_positive = layer(x_positive)
            # x_negative = layer(x_negative)
            print('training layer', i, '...')
            x_positive, x_negative = layer.forward_forward(x_positive, x_negative)


class FFDenseDiscriminator(nn.Module):
    """
    The feed forward dense discriminator.
    """

    def __init__(self, dimension, num_epoch, output_dim=10):
        super().__init__()
        self.layers = []
        self.num_epoch = num_epoch
        self.output_dim = output_dim
        for dim in range(len(dimension) - 1):
            self.layers += [FFLinearLayer(dimension[dim], dimension[dim + 1])]
            # self.layer.append(nn.Linear(dimension[dim], dimension[dim + 1]))
        # print(self.layers)

    def predict(self, x):
        goodness_score_per_label = []
        for label in range(self.output_dim):
            # perform one hot encoding#
            # print('predict label:', label, x.shape)
            encoded = overlay_y_on_x(x, label)
            goodness = []
            for idx, layer in enumerate(self.layers):
                encoded = layer(encoded)
                # print('encoded:', encoded.shape)
                # print('goodness calculation:' ,idx , encoded.pow(2).mean(1))
                goodness += [encoded.pow(2).mean(1)]
                # print('goodness:', goodness[idx])
                # print('goodness:', len(goodness), goodness[idx].shape)
            goodness_score_per_label += [sum(goodness).unsqueeze(1)]
            # print('goodness_score_per_label:', len(goodness_score_per_label), goodness_score_per_label)
        goodness_score_per_label = torch.cat(goodness_score_per_label, 1)
        # print('goodness_score_per_label:', goodness_score_per_label)
        # print('goodness_score_per_label:', goodness_score_per_label.argmax(1))
        # print('goodness_score_per_label:', goodness_score_per_label)
        return goodness_score_per_label.argmax(1)

    def train(self, x_positive, x_negative):
        for i, layer in enumerate(self.layers):
            # x_positive = layer(x_positive)
            # x_negative = layer(x_negative)
            print('training layer', i, '...')
            x_positive, x_negative = layer.forward_forward(x_positive, x_negative)
            # x_positive, x_negative = layer.train(x_positive, x_negative)


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
if __name__ == "__main__":
    # torch.manual_seed(1234)
    # dataset = CustomDisDataset('data/facades/test/')
    # XX = dataset.load_data(dataset, batch_size=32, shuffle=True)
    train_loader, test_loader = MNIST_loaders()

    # net1 = FFDenseDiscriminator([784, 784, 500, 500, 500, 500], 1000, 10)
    net = FFConvDiscriminator([1, 32, 32]).cuda()
    # xx, yy = next(iter(XX))
    # print(xx[0].shape, xx[1].shape)

    x, y = next(iter(train_loader))
    # # print(x.shape, y[0])
    x, y = x.cuda(), y.cuda()
    # print('positive input data shape:', x.shape, y.shape)
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    # print('rnd:', rnd.shape)
    x_neg = overlay_y_on_x(x, y[rnd])
    # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #     visualize_sample(data, name)
    x_pos = x_pos.reshape(-1, 1, 28, 28)
    x_neg = x_neg.reshape(-1, 1, 28, 28)
    # net1.train(x_pos, x_neg)
    # x = x.reshape(-1, 1, 28, 28)
    # print('data shape:', x.shape, y.shape)
    net.train(x_pos, x_neg)
    print('predicted train labels:', net.predict(x))
    print('train error log :', 1.0 - net.predict(x).eq(y).float().mean().item())
    print('true train labels:', y)

    # print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    # x_te = x_te.reshape(-1, 1, 28, 28)

    # select first 10 samples
    # x_te = x_te
    # y_te = y_te
    print('predicted test labels:', net.predict(x_te))
    print('test error log :', 1 - net.predict(x_te).eq(y_te).float().mean().item())
    print('true test labels:', y_te)
    # print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
    # loop over the samples
    # for i in range(10):
    #     visualize_sample(x_te[i], 'test sample')
    # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #     visualize_sample(data, name)
