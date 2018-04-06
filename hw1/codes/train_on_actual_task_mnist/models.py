import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 256, 9)
        self.pool = nn.MaxPool2d(kernel_size=6, stride=6)
        self.fc1 = nn.Linear(256 * 4 * 4, 700)
        self.final = nn.Linear(700, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
                        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.final(x)
        return x


class Medium(nn.Module):
    def __init__(self):
        super(Medium, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.final = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
                
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final(x)
        return x


class Deep(nn.Module):
    def __init__(self):
        super(Deep, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(3, 96, 5)
        self.conv2 = nn.Conv2d(96, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 3)
        
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.final = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final(x)
        return x
