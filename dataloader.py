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
