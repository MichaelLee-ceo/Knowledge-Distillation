import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, num_channel=16):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        
        self.conv2 = nn.Conv2d(num_channel, num_channel*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channel*2)

        self.conv3 = nn.Conv2d(num_channel*2, num_channel*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channel*4)
        
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(num_channel*4*4*4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.pool(x)

        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.pool(x)

        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
