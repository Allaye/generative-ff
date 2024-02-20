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



