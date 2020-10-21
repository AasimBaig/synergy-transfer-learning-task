import os
import torch
import torchvision
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets, models, transforms
import warnings
warnings.filterwarnings('ignore')


class AntsBeesDataset(Dataset):
    def __init__(self, image_path, targets, transform=None):
        self.image_path = image_path
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):

        image = Image.open(self.image_path[index])
        targets = self.targets[index]

        if self.transform:
            for transform_item in self.transform:
                image = transform_item(image)

        return (image, targets)
