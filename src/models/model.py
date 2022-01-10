import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.linear = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError('Expected input to be a 3D tensor')
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError('Expected each sample to have shape [28, 28]')
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[1]), x)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.linear(x), dim=1)

        return x
