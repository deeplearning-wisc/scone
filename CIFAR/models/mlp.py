import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from model.utils import euclidean_metric, one_hot, count_acc
from torch.autograd import Variable

#import os.path as osp

import torchvision.models as models


class woods_mlp(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #self.args = args
        hdim = 390
        # dimension: 2*14*14
        self.encoder = nn.Sequential(
            nn.Linear(28*28, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True)
        )

        self.classification = nn.Linear(hdim, num_classes)

        self.ood_fc1 = nn.Linear(hdim, 300)
        self.ood_fc2 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #import pdb
        #pdb.set_trace()
        # x.shape: [128, 1, 28, 28]
        x = x.squeeze(0)

        B, _, _, _ = x.shape
        x = x.view(x.shape[0], -1)
        # x.shape: [128, 784]
        instance_embs = self.encoder(x)

        instance_embs = torch.reshape(instance_embs, (B, -1))
        x_class = self.classification(instance_embs)

        x_ood = self.ood_fc1(instance_embs)
        x_ood = self.relu(x_ood)
        x_ood = self.ood_fc2(x_ood)

        x_all = torch.cat([x_class, x_ood], dim=1)

        return x_all







