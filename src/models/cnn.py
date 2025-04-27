import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutions (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(3, 6, 5)     # 3 channels →  6 channels
        self.conv2 = nn.Conv2d(6, 16, 5)    # 6 channels → 16 channels
        self.conv3 = nn.Conv2d(16, 32, 5)  # 16 channels → 32 channels

        # pooling
        self.pool = nn.MaxPool2d(5, 5)

        # fully connected
        self.fc1 = nn.Linear(32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

        # dropout
        self.dropout = nn.Dropout(p=0.2)
        self.dropout_conv = nn.Dropout2d(p=0.2)

    def forward(self, x):

        # convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
