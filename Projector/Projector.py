from torch import nn
import torch
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(256, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(25088, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x