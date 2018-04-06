import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
import math
import sys

sample_num = 10000
origin_x = np.linspace(0, 1, sample_num)
origin_y = (np.sin(np.pi * 5 * origin_x)) / (origin_x + 0.001)

train_x = torch.FloatTensor(origin_x)
train_x = train_x.view(-1,1)
train_y = torch.FloatTensor(origin_y)
train_y = train_y.view(-1,1)

train = torch.utils.data.TensorDataset(train_x, train_y)


neuro_num = 21
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, neuro_num)
        self.fc11 = nn.Linear(neuro_num, neuro_num)
        self.fc2 = nn.Linear(neuro_num, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        x = self.fc2(x)
        return x


test1Net = Net().cuda()
test1Net.zero_grad()

losses = []
gradNorm = []

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = optim.Adam(test1Net.parameters(), lr=learning_rate, betas = (0.9, 0.999))

if len(sys.argv) == 1:
    batch_size = 64
elif sys.argv[1] == 'full':
    batch_size = len(train)
else:
    batch_size = int(sys.argv[1])

trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

print_freq = 1
for epoch in range(200):
    running_loss = 0.0    
    for i, data in enumerate(trainloader, 0):
        inputs, y = data
        inputs, y = Variable(inputs.cuda()), Variable(y.cuda())
        optimizer.zero_grad()
        outputs = test1Net(inputs)
        loss = criterion(outputs, y)
        running_loss += loss.data[0]  
        loss.backward()
        
        if i % print_freq == print_freq - 1:
            grad_all = 0.0
            for p in test1Net.parameters():
                grad = 0.0
                if p.grad is not None:
                    grad = (p.grad.cpu().data.numpy() ** 2).sum()
                grad_all += grad
            grad = math.sqrt(grad_all)
        
            print('loss: %.3f | norm: %.3f' %(running_loss / print_freq, grad))
            losses.append(running_loss / print_freq)
            gradNorm.append(grad)
            running_loss = 0.0
            
        optimizer.step()


# plot obsevation during training
fig = plt.figure()
plt.subplot(211)
plt.plot(gradNorm[:300])
plt.ylabel('grad', fontSize=20)
plt.subplot(212)
plt.plot(losses[:300])
plt.ylabel('loss', fontSize=20)
plt.show()