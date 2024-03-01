"""
Code for research work generative forward-forward neural networks "
Date 20/02/2024.
Author: Kolade-Gideon *Allaye*
"""
import torch
import torch.nn as nn
from layers_ff import FFLinearLayer, FFConvLayer
from utils import one_hot_encode_label_on_image, overlay_y_on_x, visualize_sample
from dataloader import CustomDisDataset
from torch.utils.data import Sampler


class BaseDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass


class FFConvDiscriminator(nn.Module):
    def __init__(self, dimension, output_dim=10):
        super().__init__()
        self.layers = []
        self.output_dim = output_dim
        for dim in range(len(dimension) - 1):
            self.layers.append(FFConvLayer(dimension[dim], dimension[dim + 1], 5))
        print(self.layers)

    def predict(self, x):
        goodness_score_per_label = []
        for label in range(self.output_dim):
            # perform one hot encoding#
            # x = torch.reshape(x, (x.shape[0], -1))
            print('label:', label, x.shape)
            encoded = overlay_y_on_x(x, label)
            goodness = []
            for idx, layer in enumerate(self.layers):

                encoded = layer(encoded)
                shape = encoded.shape
                print('encoded:', encoded.shape)
                encoded = torch.reshape(encoded, (encoded.shape[0], -1))
                print('encoded:', encoded.shape)
                goodness += [encoded.pow(2).mean(1)]
                # goodness += [torch.reshape(encoded, (encoded.shape[0], -1)).pow(2).mean(1)]
                print('goodness:', len(goodness), goodness[idx].shape)
                # print('sum goodness:', sum(goodness))
                encoded = encoded.reshape(shape)
            goodness_score_per_label += [sum(goodness).unsqueeze(1)]
            print('goodness_score_per_label:', len(goodness_score_per_label))
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

    def __init__(self, dimension, num_epoch, output_dim):
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
            print('label:', label, x.shape)
            encoded = overlay_y_on_x(x, label)
            goodness = []
            for idx, layer in enumerate(self.layers):
                encoded = layer(encoded)
                print('encoded:', encoded.shape)
                goodness += [encoded.pow(2).mean(1)]
                print('goodness:', len(goodness), goodness[idx].shape)
            goodness_score_per_label += [sum(goodness).unsqueeze(1)]
        goodness_score_per_label = torch.cat(goodness_score_per_label, 1)
        return goodness_score_per_label.argmax(1)

    def train(self, x_positive, x_negative):
        for i, layer in enumerate(self.layers):
            # x_positive = layer(x_positive)
            # x_negative = layer(x_negative)
            print('training layer', i, '...')
            x_positive, x_negative = layer.forward_forward(x_positive, x_negative)


from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


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
    torch.manual_seed(1234)
    # dataset = CustomDisDataset('data/facades/test/')
    # XX = dataset.load_data(dataset, batch_size=32, shuffle=True)
    train_loader, test_loader = MNIST_loaders()

    net1 = FFDenseDiscriminator([784, 500, 500], 100, 10)
    net = FFConvDiscriminator([1, 6, 16, 120]).cuda()
    # xx, yy = next(iter(XX))
    # print(xx[0].shape, xx[1].shape)

    x, y = next(iter(train_loader))
    # # print(x.shape, y[0])
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    # for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #     visualize_sample(data, name)
    x_pos = x_pos.reshape(-1, 1, 28, 28)
    x_neg = x_neg.reshape(-1, 1, 28, 28)
    # net.train(x_pos, x_neg)
    x = x.reshape(-1, 1, 28, 28)
    # print('data shape:', x.shape, y.shape)
    # print('train error:', 1.0 - net1.predict(x).eq(y).float().mean().item())
    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    # x_te, y_te = next(iter(test_loader))
    # x_te, y_te = x_te.cuda(), y_te.cuda()
    # x_te = x_te.reshape(-1, 1, 28, 28)
    # print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())

    #
    # for data, name in zip([xx, yy], ['positive', 'negative']):
    #     visualize_sample(data, name)
    # x_pos = x_pos.reshape(-1, 1, 28, 28)
    # x_neg = x_neg.reshape(-1, 1, 28, 28)
    # x_neg = x_neg.to(device)
    #   x
    #     # def overlay_y_on_x(x, y):
    #     #     """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    #     #     """
    #     #     x_ = x.clone()
    #     #     x_[:, :10] *= 0.0
    #     #     x_[range(x.shape[0]), y] = x.max()
    #     #     return x_
    #     #
    #     # def visualize_sample(data, name='', idx=0):
    #     #     reshaped = data[idx].cpu().reshape(28, 28)
    #     #     plt.figure(figsize=(4, 4))
    #     #     plt.title(name)
    #     #     plt.imshow(reshaped, cmap="gray")
    #     #     plt.show()
    #
    # print('data shape,', x_pos.shape, x_neg.shape)
    # net.train(x_pos, x_neg)
#
#     print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())
#
#     x_te, y_te = next(iter(test_loader))
#     x_te, y_te = x_te.cuda(), y_te.cuda()
#
#     print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
#
# # from torch.nn import functional as F
# #
# # F.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3)
# net = FFDenseDiscriminator([784, 784, 500, 500, 500, 784, 500], 100, 10)
# # train error: 0.7913400083780289
# # test error: 0.789400011301040
#
# # net = FFDenseDiscriminator([784, 500, 500], 100, 10)
# # train error: 0.8306400030851364
# # test error: 0.8275000005960464

# net = FFConvDiscriminator([3, 6, 16, 120])
