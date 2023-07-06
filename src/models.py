import torch
import torch.nn as nn
from modules import Conv_2d
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Sequential(nn.Linear(128 * 32, 256), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class FCN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FCN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=260, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 120)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x