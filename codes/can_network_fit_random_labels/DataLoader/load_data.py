
# coding: utf-8

# # import module

# In[1]:


import torch
import torchvision
from torchvision import datasets, transforms
import argparse


# # define load data function

# In[2]:

"""
def load_data(args):
    #Fetch MNIST dataset
    kwargs = {}
    
    #Fetch training data
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, 
                            transform=transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    
    #Fetch testing data
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.batch_size, shuffle=True, **kwargs)

    return (train_loader, test_loader)
"""
def load_data(args):
    transform = transforms.Compose([transforms.ToTensor(), \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../cifar10_data', \
        train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
        shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../cifar10_data', \
        train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, \
        shuffle=False, num_workers=2)

    return [trainloader, testloader]

