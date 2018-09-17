#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""


import torch.nn as nn
import torch.nn.functional as F
# Raw waveform based Convolution neural networks
class ConvNet(nn.Module):
    def __init__(self, num_classes=11):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=5, stride=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=5, stride=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(60, 60, kernel_size=5, stride=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))


        self.fc1 = nn.Linear(7440, 5000)
        self.fc2 = nn.Linear(5000,1000)
        self.fc3 = nn.Linear(1000, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = F.dropout(F.relu(self.fc1(out)),p=0.3)
        out = F.dropout(F.relu(self.fc2(out)),p=0.3)
        out = self.fc3(out)
        return out

        out = self.fc3(out)
        return out

