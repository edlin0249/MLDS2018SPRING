
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=2)


import matplotlib.pyplot as plt
import numpy as np


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net_DNN1(nn.Module):
    def __init__(self):
        super(Net_DNN1, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(28*28, 10000)
        self.fc2 = nn.Linear(10000, 10)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
    
class Net_DNN2(nn.Module):
    def __init__(self):
        super(Net_DNN2, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(28*28, 1000)
        self.fc2 = nn.Linear(1000, 5000)
        self.fc3 = nn.Linear(5000, 400)
        self.fc4 = nn.Linear(400, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Net_DNN3(nn.Module):
    def __init__(self):
        super(Net_DNN3, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(28*28, 1300)
        self.fc2 = nn.Linear(1300, 1300)
        self.fc3 = nn.Linear(1300, 1300)
        self.fc4 = nn.Linear(1300, 1300)
        self.fc5 = nn.Linear(1300, 1300)
        self.fc6 = nn.Linear(1300, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


#net = Net()
net_dnn3 = Net_DNN3()


print(net_dnn3)

#nets = [net_dnn_hidden0_dim128, net_dnn_hidden0_dim200, net_dnn_hidden0_dim260, net_dnn_hidden4_dim128, net_dnn_hidden8_dim128]
parrsum=0
for param in net_dnn3.parameters():
    parrsum += param.nelement()
print("how many parameters: %d" % parrsum)

# In[ ]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net_dnn_hidden0_dim20.parameters(), lr=0.01, betas=(0.9, 0.99))
optimizer = optim.SGD(net_dnn3.parameters(), lr=0.001, momentum=0.9)

print("optimizer is finished")

l_hist_train = []
acc_hist_train = []
l_hist_test = []
acc_hist_test = []

# In[ ]:
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    total_size = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net_dnn3(inputs)
        

        _, predicted = torch.max(outputs.data, 1)
        
        total_size += labels.data.size(0)
        #print("total_size = %d" % total_size)

        correct += (predicted == labels.data).sum()
        #print("correct = %d" % (predicted == labels.data).sum())

        #print(type(outputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0] * labels.data.size(0)
        
    print("epoch : %d, train_loss = %f" % (epoch+1, running_loss / total_size))
    print("epoch : %d, train_acc = %f" % (epoch+1, correct / total_size))
    l_hist_train.append(running_loss / total_size)
    acc_hist_train.append(correct / total_size)


    running_loss = 0.0
    total_size = 0
    correct = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net_dnn3(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_size += labels.data.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        running_loss += loss.data[0] * labels.data.size(0)

    print("epoch : %d, test_loss = %f" % (epoch+1, running_loss / total_size))
    print("epoch : %d, test_acc = %f" % (epoch+1, correct / total_size))
    l_hist_test.append(running_loss / total_size)
    acc_hist_test.append(correct / total_size)



print('Finished Training')



# In[ ]:


plt.plot(l_hist_train)

plt.show()

plt.plot(acc_hist_train)

plt.show()

plt.plot(l_hist_test)

plt.show()

plt.plot(acc_hist_test)

plt.show()


#output loss history to csv file

with open("net_dnn3_loss_train.csv", "w") as F:
    for t in l_hist_train:
        F.write(str(t)+",")
    F.write("0")


with open("net_dnn3_acc_train.csv", "w") as F:
    for t in acc_hist_train:
        F.write(str(t)+",")
    F.write("0")

with open("net_dnn3_loss_test.csv", "w") as F:
    for t in l_hist_test:
        F.write(str(t)+",")
    F.write("0")

with open("net_dnn3_acc_test.csv", "w") as F:
    for t in acc_hist_test:
        F.write(str(t)+",")
    F.write("0")