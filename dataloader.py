import tarfile
import os
import torch
import cv2
import glob
from utils import split_image
from torch.utils.data import DataLoader, Dataset

path = "data/facades"


class CustomDisDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = None
        self.label = None
        self.data_path = glob.glob(path)
        # self.data, self.label = self.split_image(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_list = []
        for f in glob.glob(self.data_path + '/*.jpg'):
            image_list.append(f)

        for idx in range(len(image_list)):
            image = cv2.imread(image_list[idx])
            data, label = split_image(image)
        return self.data, self.label

# dataloader = CustomDisDataset('dstas/datset')

# for data, label in dataloader:
=======
# note for the dataloader, the data should be in the shape of (batch_size, channels, height, width), [row, label]
# the label will be a list of integers representing the class of the data
# index 1 of the data will have a label of 1 and so on
# so if the data contains 2000 samples, the label will be a list of 2000 integers
