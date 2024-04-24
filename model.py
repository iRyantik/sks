from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import copy
import torch.nn as nn
import torch.nn.functional as F

# Implement a convolutional neural network (https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)

'''
torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

torch.nn.MaxPool2d(kernel_size, stride=None, 
    padding=0, dilation=1, return_indices=False, ceil_mode=False)

kernel_size: square window of size (i.e. 3)

torch.nn.Linear(in_features, out_features, 
    bias=True)


'''

class Net(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            # [4, 3, 32, 32]
            nn.Conv2d(3, 6, kernel_size=5, padding=2), # (32-5+4)/1 + 1 = 32
            nn.Conv2d(6, 12, kernel_size=5, padding=0), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14
            
            nn.Conv2d(12, 24, kernel_size=3, padding=1), #(14-3+2)/1 +1 = 14
            nn.Conv2d(24, 32, kernel_size=3, padding=0), # 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 6 [4, 32, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, img):
        x = self.features(img)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet, self).__init__()
        # [4, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 6, 5) #C_in=3, C_out=6, Kernel=5
        # (32 - 5)/1+1 = 28
        # [4, 6, 28, 28] > reLU
        self.pool = nn.MaxPool2d(2, 2) #kernel=2, stride=2
        # (28-2)/2 + 1 = 14
        # [4, 6, 14, 14]
        self.conv2 = nn.Conv2d(6, 16, 5)
        # (14-5)/1 + 1 = 10 
        # [4, 16, 10, 10] > reLU > MaxPooling > [4, 16, 5, 5]
        # 400 = 16 * 5 * 5
        # flattern > [4, 400] 400 = 16 * 5 * 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 400 > 120 > 84 > 10

    def forward(self, img):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x