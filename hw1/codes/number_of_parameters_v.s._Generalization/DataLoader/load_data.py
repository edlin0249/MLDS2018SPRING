import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def load_data_train(args):
    transform = transforms.Compose([transforms.ToTensor(), \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../cifar10_data', \
        train=True, download=True, transform=transform)

    validset = torchvision.datasets.CIFAR10(root='../cifar10_data', train=True,  \
        download=True, transform=transform)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(args.valid_size * num_train))

    if args.is_shuffle:
        random_seed = 0
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
        sampler=train_sampler, num_workers=2)

    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, \
        sampler=valid_sampler, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../cifar10_data', \
        train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, \
        shuffle=False, num_workers=2)

    return [trainloader, validloader, testloader]

def load_data_test(args):
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