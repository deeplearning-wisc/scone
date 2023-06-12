import numpy as np
import torch
import torch.utils.data as data
import torchvision

from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from random import sample, random

class CorMNISTDataset(data.Dataset):
    def __init__(self, set_name, data_path):
        self._image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307,], [0.3081,])
        ])

        images = np.load(os.path.join(data_path, 'dotted_line', set_name + '_images.npy'))
        labels = np.load(os.path.join(data_path, 'dotted_line', set_name + '_labels.npy'))

        self.data = images
        self.label = labels

        self.num_class = 10

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self._image_transformer(img)
        
        return img, label

    def __len__(self):
        return len(self.data)










