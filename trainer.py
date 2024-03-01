import cv2
import torch
from discriminators_ff import *
from utils import *
from dataloader import CustomDisDataset

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    torch.manual_seed(1234)
    dataset = CustomDisDataset('data/facades/test/')
    data = dataset.load_data(dataset, batch_size=32, shuffle=True)
    train_loader, test_loader = MNIST_loaders()
    # net = FFConvDiscriminator([1, 6, 16, 120]).cuda()
    real, fake = next(iter(data))
    X, Y = next(iter(train_loader))
    x, y = X.cuda(), Y.cuda()
    x_pos = overlay_y_on_x(x, y)
    real_label, fake_label = generate_label(real, 1), generate_label(fake, 0)
    # real, fake = real.cuda(), fake.cuda()
    # real_label, fake_label = real_label.cuda(), fake_label.cuda()
    positive_real = overlay_y_on_x1(real, real_label)
    positive_fake = overlay_y_on_x1(fake, fake_label)
    # merge the positive_real and positive_fake into one tensor called positive data
    # print(positive_real.shape, positive_fake.shape)
    # positive_data = torch.concat([positive_real, positive_fake], 0)
    data = positive_fake[10].permute(1, 2, 0)
    reshaped = data.cpu().reshape(256, 256, 3)# .numpy()
    # reshaped = cv.cvtColor(reshaped, cv2.COLOR_BGR2GRAY)
    #
    plt.figure(figsize=(4, 4))
    plt.subplot(1, 2, 1)
    plt.title('name')
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    # print(x_pos[0].shape, x_pos.shape)
    reshaped = x_pos[0].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title('name')
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    # for data, name in zip([fake, real, positive_real, positive_fake], ['fake', 'real', 'positive', 'negative']):
    #     # visualize_sample(data, name)
    #     data = data[10].permute(1, 2, 0)
    #     reshaped = data.cpu().reshape(256, 256, 3).numpy()
    #     #
    #     # # plt.figure(figsize=(5, 5))
    #     # plt.subplot(1, 2, 1)
    #     plt.title(name)
    #     plt.imshow(cv.cvtColor(reshaped, cv.COLOR_BGR2RGB))
    #     plt.show()
