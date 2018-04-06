
# coding: utf-8

# In[11]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np


# In[12]:


def par21d(model):
    par1d_all = np.array([[]])
    for par in model.parameters():
        par_1d = np.reshape((par.cpu().data.numpy()),(1,-1))
        par1d_all = np.concatenate((par1d_all,par_1d), axis = 1)
    return par1d_all


# In[13]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data_mnist', train=False,
                                       download=True, transform=transform)


# In[14]:


gain = 10
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #torch.cuda.manual_seed(20)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)
        self.fc1 = nn.Linear(4*4*50, 500)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)
        self.fc2 = nn.Linear(500, 10)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"


# In[15]:


#torch.manual_seed(100)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=-1,
                                         shuffle=False, num_workers=2)

#trainloader_t = torch.utils.data.DataLoader(trainset, batch_size=1000,shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(model1.parameters(), lr = 0.1)


# In[20]:


parVar = []
parLoss = []
model_label = []
mlabel = 0


# In[21]:


losses = []
gradNorm = []


# In[22]:


freq = 2
batches = [64,2000,512,128]
epoches = [10,10,10,10]
#epoches = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

for e,epoch in enumerate(epoches):  # loop over the dataset multiple times
    #torch.manual_seed(20)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batches[e],
                                          shuffle=True, num_workers=2)
    model1 = LeNet()
    model1.cuda()
    optimizer = optim.Adam(model1.parameters(), lr = 0.001)
    
    mlabel+=1
       
    for _ in range(epoch):
        running_loss = 0.0
        epoch_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model1(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.data[0]
            epoch_loss += loss.data[0]
            losses.append(loss.data[0])
            
            loss.backward()
            optimizer.step()
            
            
            
            if i % freq == freq - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      ( _ + 1, i + 1, running_loss / freq))
                running_loss = 0.0
 
            
        
        t_loss = epoch_loss / (i+1)
        losses.append(t_loss)
        
        if( _ % 3 == 0):
            par1d_all = par21d(model1) 
            parVar.append(par1d_all)
            parLoss.append(round(t_loss, 3))
            model_label.append(mlabel)        
       
        
    #torch.save(model1.state_dict(), text + str(batches[e]))
print('Finished Training')


# In[24]:


# plot obsevation during training
plt.subplot(211)
plt.plot(losses[:0])
plt.subplot(212)
plt.plot(gradNorm[:0])
plt.show()


# In[28]:


paras_num = len(parVar)
matrix_grad = np.array(parVar)
matrix_grad = matrix_grad.reshape((paras_num, -1))

from sklearn.decomposition import PCA
co = np.array(model_label)
par_losses = np.array(parLoss) 
#matrix_grad = matrix_grad[(co != 70) * (co != 8) * (par_losses<1000)]

pca = PCA(n_components = 2)
x_r = pca.fit(matrix_grad).transform(matrix_grad)
print(x_r)
#co = co[(co != 70) * (co != 8) * (par_losses<1000)]
plt.scatter(x_r[:, 0], x_r[:, 1], c = co * 50.0, alpha = 0.6, s = (par_losses + 0.001) * 1000)

for i in range(paras_num):
    plt.text(x_r[i, 0], x_r[i, 1], par_losses[i], color = 'red')
    pass
plt.show()

