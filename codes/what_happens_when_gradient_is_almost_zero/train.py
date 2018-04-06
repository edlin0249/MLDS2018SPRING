import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
import pandas as pd
import sys

def propogate_loss(net, train):
    trainloader = torch.utils.data.DataLoader(train, batch_size=len(list(train)), shuffle=True)
    inputs, y = list(trainloader)[0]
    
    criterion = nn.MSELoss()
    out = net(Variable(inputs.cuda()))
    loss = criterion(out, Variable(y.cuda()))
    
    return loss


def compute_grad_norm(model, loss):
    model.zero_grad()
    loss.backward()
    
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    return grad_all ** 0.5


def update_parameters(net, change):
    pivot = 0
    for layer in net.parameters():
        flat = layer.view(-1).data
        flat.add_(change[pivot:pivot + len(flat)])
        pivot += len(flat)


def compute_parameters_num(net):
    parrsum=0
    for param in net.parameters():
        parrsum += param.nelement()
    return parrsum


sample_num = 10000
origin_x = np.linspace(0, 1, sample_num)
origin_y = (np.sin(np.pi * 5 * origin_x)) / (origin_x + 0.001)

train_x = torch.FloatTensor(origin_x)
train_x = train_x.view(-1,1)
train_y = torch.FloatTensor(origin_y)
train_y = train_y.view(-1,1)

train = torch.utils.data.TensorDataset(train_x, train_y)


neuro_num = 3

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

def train_loss(net, train, losses=[], gradNorm=[]):
    print('train by loss')
    train_epoch = 50
    print_freq = 200   
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas = (0.9, 0.999))
    trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    
    for epoch in range(train_epoch):
        running_loss = 0.0    
        for i, data in enumerate(trainloader, 0):
            inputs, y = data
            inputs, y = Variable(inputs.cuda()), Variable(y.cuda())
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, y)
            running_loss += loss.data[0]  

            if i % print_freq == print_freq - 1:
                grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()

                print('loss: %.3f | norm: %.3f' %(running_loss / print_freq, grad_norm.data[0]))
                losses.append(running_loss / print_freq)
                gradNorm.append(grad_norm.data[0])
                running_loss = 0.0

            loss.backward()  
            optimizer.step()
    
    return losses, gradNorm


def train_norm(net, train, losses=[], gradNorm=[]):
    print('train by norm')
    train_epoch = 400
    print_freq = 20  
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas = (0.9, 0.999))
    trainloader = torch.utils.data.DataLoader(train, batch_size=len(list(train)), shuffle=True)
    
    for epoch in range(train_epoch):
        inputs, y = list(trainloader)[0]
        inputs, y = Variable(inputs.cuda()), Variable(y.cuda())
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, y)

        grad_params = torch.autograd.grad(loss, net.parameters(), create_graph=True)

        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()

        if epoch % print_freq == print_freq - 1:
            print('loss: %.3f | norm: %.3f' %(loss.data[0], grad_norm.data[0]))
            losses.append(loss.data[0])
            gradNorm.append(grad_norm.data[0])

        grad_norm.backward()    
        optimizer.step()
    
    return losses, gradNorm


def hessian(net, loss):
    one_grads = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    
    one_grads_flat = ()
    for one_grad in one_grads:
        one_grads_flat += one_grad.view(-1).split(1)
    
    parrsum = compute_parameters_num(net)
    matrix = torch.ones(parrsum, parrsum).cuda()
    for i, one_grad in enumerate(one_grads_flat):
        grads = torch.autograd.grad(one_grad, net.parameters(), create_graph=True)
        pivot = 0
        for grad in grads:
            tmp = grad.contiguous().view(-1).data
            matrix[i, pivot : pivot + len(tmp)] = tmp
            pivot += len(tmp)
    
    symmetric = matrix.tril() + matrix.tril(diagonal=-1).t()
    
    return symmetric, torch.cat(one_grads_flat).data


def train_hessian(net, train, losses=[], gradNorm=[]):
    print('train by hessian')
#     train_epoch = 20
    
    lr = 2
    last = 1e8
    while True:
        loss = propogate_loss(net, train)
        losses.append(loss.data[0])
        
        loss = propogate_loss(net, train)
        matrix, grad = hessian(net, loss)
        grad_norm = np.sqrt((grad ** 2).sum())
        gradNorm.append(grad_norm)
        
        if loss.data[0] > last:
            lr *= 2
        else:
            if lr > 1.5:
                lr -= 1
            elif lr > 0.2:
                lr -= 0.1
            else:
                lr = 0.1
                
        last = loss.data[0]
#         lr = grad_norm
        
        print('loss: %.3f | norm: %.3f | step: %f' %(loss.data[0], grad_norm, lr))
        
        change = torch.mm((matrix + torch.eye(matrix.size()[0]).cuda() * lr).inverse(), grad.view(-1, 1)).view(-1)
        update_parameters(net, -change)
        
        if grad_norm < 0.008:
            break
    
    return losses, gradNorm


alllosses = []
allgradNorm = []
allnets = []
allratio = []

model_num = int(sys.argv[1])
update_type = sys.argv[2]
if update_type == 'GN':
    find_critical_point = train_norm
elif update_type == 'LM':
    find_critical_point = train_hessian
else:
    print('ERROR, PLEASE FOLLOW THE CORRECT COMMAND FORMAT')

num = 1
while num <= model_num:
    net = Net().cuda()
    losses = []
    gradNorm = []
    
    losses, gradNorm = train_loss(net, train, losses, gradNorm)
    losses, gradNorm = find_critical_point(net, train, losses, gradNorm)

    loss = propogate_loss(net, train)
    if compute_grad_norm(net, loss) > 0.04:
        print('NET' + str(num) + ' AGAIN')
        continue


    loss = propogate_loss(net, train)
    matrix, _ = hessian(net, loss)
    eigs = matrix.cpu().symeig()[0]
    ratio = (eigs > 0).sum() / len(eigs)
    
    print('NET' + str(num) + ' | loss: ' + str(losses[-1]) + ', ratio: ' + str(ratio))
    
    alllosses.append(losses[-1])
    allgradNorm.append(gradNorm[-1])
    allnets.append(net)
    allratio.append(ratio)
    num += 1


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('minimum ratio', fontSize=20)
ax.set_ylabel('loss', fontSize=20)
plt.plot(np.array(allratio), np.array(alllosses), 'ro')
plt.show()
