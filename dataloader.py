import tarfile
import os

import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, PILToTensor
from PIL import Image
from utils import split_image


class CustomDataset(Dataset):
    def __init__(self, path, transform=[ToTensor()]):
        self.filenames = os.listdir(path)
        self.transform = Compose(transform)
        self.file_path = path
        self.full_image_path = []
        for file in self.filenames:
            self.full_image_path.append(os.path.join(self.file_path, file))

    def __len__(self):
        """
        Get the length of the dataset
        :return:
        """
        return len(self.full_image_path)

    def __getitem__(self, idx):
        """
        Get the data by the index
        :param idx:
        :return:
        """
        image = np.array(Image.open(self.full_image_path[idx]))
        print(image.shape)
        x, y = split_image(image)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    @staticmethod
    def load_data(dataset_object, batch_size=32, shuffle=True):
        """
         Load the data from the file system
        :param dataset_object:
        :param batch_size: the batch size
        :param shuffle: whether to shuffle the data or not
        :return: the data loader
        """
        return DataLoader(dataset_object, batch_size=batch_size, shuffle=shuffle)


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)


# note for the dataloader, the data should be in the shape of (batch_size, channels, height, width), [row, label]
# the label will be a list of integers representing the class of the data
# index 1 of the data will have a label of 1 and so on
# so if the data contains 2000 samples, the label will be a list of 2000 integers

# dataset = CustomDisDataset('data/facades/test/')
# dataloader = dataset.load_data(dataset, batch_size=32, shuffle=True)
# for i, (data, label) in enumerate(dataloader):
#     print(data.shape, label.shape)
#     # plt.imshow(data[0].reshape(256, 256, 3))
#     # plt.show(block=True)
#     if i == 10:
#         break
