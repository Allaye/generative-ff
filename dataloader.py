import tarfile
import os
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDisDataset(Dataset):
    def __init__(self, data, labels, cutsize, transform=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# note for the dataloader, the data should be in the shape of (batch_size, channels, height, width), [row, label]
# the label will be a list of integers representing the class of the data
# index 1 of the data will have a label of 1 and so on
# so if the data contains 2000 samples, the label will be a list of 2000 integers
