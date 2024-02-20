import torch
import torch.nn as nn
from layers import FFLinearLayer


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
    def __init__(self, dimension, num_epoch):
        super().__init__()
        self.layer = []
        for dim in range(len(dimension) - 1):
            self.layer += [FFLinearLayer(dimension[dim], dimension[dim + 1])]
            # self.layer.append(nn.Linear(dimension[dim], dimension[dim + 1]))

    def predict(self, x):
        pass
