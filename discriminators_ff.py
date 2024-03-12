"""
Code for research work for the discriminator forward-forward neural networks "
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

    def layer_train(self, x_positive, x_negative, tra=False, epoch=None, logger=None):
        for i, layer in enumerate(self.layers):
            # x_positive = layer(x_positive)
            # x_negative = layer(x_negative)
            print('training layer', i, '...')
            loss, x_positive, x_negative = layer.forward_forward_trad(x_positive, x_negative)
            if epoch % 20 == 0:
                # print('epoch:', epoch, 'loss:', loss)
                logger.log({'epoch': epoch, 'loss': loss})

            # x_positive, x_negative = layer.train(x_positive, x_negative)

    def train(self, x_positive, x_negative, logger=None, model=None, x=None, y=None):
        for i in range(self.num_epoch):
            self.layer_train(x_positive, x_negative, epoch=i, logger=logger)
            train_loss = 1.0 - model.predict(x).eq(y).float().mean().item()
            logger.log({'epoch': i, 'train_loss': train_loss})
            print('epoch:', i, '...')
        print('training completed...')
