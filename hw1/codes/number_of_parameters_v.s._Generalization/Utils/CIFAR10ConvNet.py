from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 


class CIFAR10ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10ConvNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(3*32*32, 2000)
        self.fc2 = nn.Linear(2000, 5000)
        self.fc3 = nn.Linear(5000, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

