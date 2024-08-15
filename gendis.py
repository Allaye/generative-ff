import torch
from torch import nn
from layers_ff import FFLinearLayer




# data = torch.randn(10, 784).to('cuda')
# net = FFLinearLayer(784, 10).to('cuda')
# print(data.shape)
# print(net(data).shape)


class FFLinearGen(nn.Module):
    """
    the Generative Network takes a latent variable vector as input,
    and returns a 784 valued vector, which corresponds to a flattened 28x28 image.
    Args:
        in_features: The size of the input feature
        out_features: The size of the output feature
    """
    def __init__(self, in_features=100, out_features=784):
        super(FFLinearGen, self).__init__()

        self.down0 = nn.Sequential(
            FFLinearLayer(in_features, 256, act=nn.LeakyReLU)
        )
        self.down1 = nn.Sequential(
            FFLinearLayer(256, 512, act=nn.LeakyReLU)
        )
        self.down2 = nn.Sequential(
            FFLinearLayer(512, 1024, act=nn.LeakyReLU)
        )
        self.down3 = nn.Sequential(
            FFLinearLayer(1024, out_features, act=nn.Tanh)
        )
    def forward(self, x):
        x = self.down0(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class FFLinearDis(nn.Module):
    """
     This network will take a flattened image as its input, and return the probability of
     it belonging to the real dataset, or the synthetic dataset.
     The input size for each image will be 28x28=784 pixels, and the output will be a single scalar number.
     Args:
            in_features: The size of the input feature
            out_features: The size of the output feature
    """
    def __init__(self, in_features=784, out_features=1):
        super(FFLinearDis, self).__init__()

        self.up0 = nn.Sequential(
            FFLinearLayer(784, 1024, act=nn.LeakyReLU, drop=True, drop_rate=0.3)
        )
        self.up1 = nn.Sequential(
            FFLinearLayer(1024, 512, act=nn.LeakyReLU, drop=True, drop_rate=0.3)
        )
        self.up2 = nn.Sequential(
            FFLinearLayer(512, 256, act=nn.LeakyReLU, drop=True, drop_rate=0.3)
        )
        self.up3 = nn.Sequential(
            FFLinearLayer(256, 1, act=nn.Sigmoid)
        )

    def forward(self, x):
        x = self.up0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x






gen = FFLinearGen().to('cuda')
dis = FFLinearDis().to('cuda')
data = torch.randn(10, 100).to('cuda')
print('Gen shape', gen)
print('Dis shape', dis)
print('data shape', data.shape)
print('gen output shape', gen(data).shape)
print('dis output shape', dis(gen(data)).shape)
