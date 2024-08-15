import torch
from torch import nn
from layers_ff import FFLinearLayer




# data = torch.randn(10, 784).to('cuda')
# net = FFLinearLayer(784, 10).to('cuda')
# print(data.shape)
# print(net(data).shape)


class FFLinearGen(nn.Module):
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

class FFLinearDis():
    pass

gen = FFLinearGen().to('cuda')
data = torch.randn(10, 100).to('cuda')
print('Gen shape', gen)
print('data shape', data.shape)
print('gen output shape', gen(data).shape)