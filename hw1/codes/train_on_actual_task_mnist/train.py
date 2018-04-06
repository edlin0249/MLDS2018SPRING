import os, sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import models
import importlib
importlib.reload(models)

def main():
    # get data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])

    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    losses = []
    accs = []

    for model_type in range(3):
        print('Start training model' + str(model_type))

        if model_type == 0:
            net = models.Shallow().cuda()
        elif model_type == 1:
            net = models.Medium().cuda()
        elif model_type == 2:
            net = models.Deep().cuda()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        train_loss = []
        accuracy = []

        train_num = len(trainset)
        print_freq = 32
        epochs = 50

        for epoch in range(epochs):  # loop over the dataset multiple times
            print('EPOCH ' + str(epoch))
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # calculate loss and accuracy
                running_loss += loss.data[0]
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.data).sum()
                total += labels.size(0)
                    
                # print statistics
                if i % print_freq == print_freq - 1:
                    persentage = (i + 1) * batch_size / train_num
                    avg_loss = running_loss / print_freq
                    avg_acc = correct / total
                    
                    print('%.3f | loss: %.10f | acc: %.10f' % (persentage, avg_loss, avg_acc))
                    train_loss.append(avg_loss)
                    accuracy.append(avg_acc)
                    
                    running_loss = 0
                    correct = 0
                    total = 0

        losses.append(train_loss.copy())
        accs.append(accuracy.copy())

    fg, ax = plt.subplots()
    ax.plot(losses[0], label='Shallow')
    ax.plot(losses[1], label='Medium')
    ax.plot(losses[2], label='Deep')
    ax.set_xlabel('Epoch', size=18)
    ax.set_ylabel('Loss', size = 18)
    ax.legend(loc=0, prop={'size':15})
    plt.show()

    fg, ax = plt.subplots()
    ax.plot(accs[0], label='Shallow')
    ax.plot(accs[1], label='Medium')
    ax.plot(accs[2], label='Deep')
    ax.set_xlabel('Epoch', size=18)
    ax.set_ylabel('Accuracy', size = 18)
    ax.legend(loc=0, prop={'size':15})
    plt.show()


if __name__ == '__main__':
    main()