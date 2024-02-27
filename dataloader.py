import tarfile
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils import overlay_y_on_x

class CustomDisDataset(Dataset):
    def __init__(self, path, transform=None):
        self.filenames = os.listdir(path)
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

        image = Image.open(self.full_image_path[idx])
        if self.transform:
            image = self.transform(image)
        x, y =
        return self.data[idx], self.labels[idx]


# note for the dataloader, the data should be in the shape of (batch_size, channels, height, width), [row, label]
# the label will be a list of integers representing the class of the data
# index 1 of the data will have a label of 1 and so on
# so if the data contains 2000 samples, the label will be a list of 2000 integers


paath = os.listdir('data/facades/test')
print(paath)
