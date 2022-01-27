import torch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetMadry(nn.Module):

    def __init__(self, num_classes=10, feature_extractor=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.features(x)

        if self.feature_extractor:
            return x

        return self.fc2(x)

    def features(self, x, return_acts=False):
        a1 = self.conv1(x)
        h1 = F.relu(a1)
        h1 = F.max_pool2d(h1, 2, 2)
        a2 = self.conv2(h1)
        h2 = F.relu(a2)
        h2 = F.max_pool2d(h2, 2, 2)
        h2 = self.flatten(h2)
        a3 = self.fc1(h2)
        h3 = F.relu(a3)
        return h3
