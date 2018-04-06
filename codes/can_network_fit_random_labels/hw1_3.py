
# coding: utf-8

# # import module

# In[15]:


import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from DataLoader.load_data import load_data
from Utils.model import CIFAR10ConvNet
import numpy as np 
import matplotlib.pyplot as plt
from pdb import set_trace

PATH = './Model/final_model'
# # define train function

# In[16]:


def train(args, train_loader, model, idx):
    model.train()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #for i in range(args.epochs):
    running_total = 0
    running_correct = 0
    running_loss = 0

    running_total_t = 0
    running_correct_t = 0
    running_loss_t = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        correct = (predicted == target.data).sum()
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss_t += loss.data[0] * target.size(0)
        running_total_t += total
        running_correct_t += correct
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}  ({:.0f}%)]\tLoss: {:.6f} \tAccuracy: {:.6f}%'.format(idx, batch_idx*len(data), 
                50000, 100. * batch_idx / len(train_loader), running_loss_t / 1000, 100 * running_correct_t / running_total_t))

            running_loss_t = 0
            #_, predicted = torch.max(output.data, 1)
            running_total_t = 0
            running_correct_t = 0

        running_loss += loss.data[0] * target.size(0)
        #_, predicted = torch.max(output.data, 1)
        running_total += total
        running_correct += correct

    
    all_loss = running_loss / running_total
    all_accuracy = running_correct / running_total
    return [all_loss, all_accuracy]
        
    
    
    
    


# # define test function

# In[17]:


def test(args, train_loader, test_loader, model, idx):
    model.eval()
   
    train_loss = 0
    correct = 0
    total = 0
    loss_func = torch.nn.CrossEntropyLoss()
    for inputs, labels in train_loader:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.data[0] * labels.size(0)
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()      
    print("epochs : %d, train_loss = %f, train_acc = %f" % (idx, train_loss / total, correct / total))
    train_loss = train_loss / total

    
    test_loss = 0
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        test_loss += loss.data[0] * labels.size(0)
        #pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()  
    print("epochs : %d, test_loss = %f, test_acc = %f" % (idx, test_loss / total, correct / total))
    test_loss = test_loss / total

    return [train_loss, test_loss]





# # define main function

# In[24]:


def main():
    parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
    parser.add_argument('action', default='train', choices=['train','test'], help='select one of actions(train/test)')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size(default=100)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='learning rate(default=0.1)')
    parser.add_argument('--epochs', type=int, default=1, help='epochs(default=200)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum(default=0.5)')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    args = parser.parse_args()
    #parser.parse_args('python hw1_3.py train'.split())
    train_loader, test_loader = load_data(args)
    train_loader_ds = []
    for data, target in train_loader:
        #print("before shuffling:")
        #print(target)
        #print('type(data) = ', type(data))
        #print('type(target) = ', type(target))
        data_t = np.array(data.numpy())
        target_t = np.array(target.numpy())
        np.random.seed(0)
        np.random.shuffle(target_t)
        train_loader_ds.append([torch.FloatTensor(data_t), torch.LongTensor(target_t)])
        #print("after shuffling:")
        #print(target)
    #train_loader_ds = np.array(train_loader_ds)
    #train_data_size = len(train_loader.dataset)


    if args.action == 'train':
        model = CIFAR10ConvNet().cuda()
        print(model)
        loss_hist = []
        accuracy_hist = []
        #how many parameters in Network
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('total parameters = ', pytorch_total_params)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('total trainable parameters = ', pytorch_total_params)
        train_loss_hist = []
        test_loss_hist = []
        for i in range(args.epochs):
            all_loss, all_accuracy = train(args, train_loader_ds, model, idx=i)
            torch.save(model, PATH + "_" + str(i))
            loss_hist.append(all_loss)
            accuracy_hist.append(all_accuracy)

            train_loss, test_loss = test(args, train_loader_ds, test_loader, model, idx=i)
            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)

        set_trace()
        ax = plt.subplot()
        ax.plot(train_loss_hist, 'bo', label='train_loss')
        ax.plot(test_loss_hist, 'yo', label='test_loss')
        ax.legend(loc=0)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('model loss')

        plt.show()

        """
        plot_loss, = plt.plot(loss_hist, label='loss')
        plot_acc, = plt.plot(accuracy_hist, label='accuracy')

        plt.legend(handles=[plot_loss, plot_acc], loc='best')

        plt.show()
        """

    elif args.action == 'test':
        train_loss_hist = []
        test_loss_hist = []
        for i in range(args.epochs):
            model = torch.load(PATH+"_"+str(i))
            train_loss, test_loss = test(args, train_loader_ds, test_loader, model, idx=i)
            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)
        train_loss_plot, = plt.plot(train_loss_hist, label='train_loss')
        test_loss_plot, = plt.plot(test_loss_hist, label='test_loss')
        plt.legend(handles=[train_loss, test_loss], loc='best')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('model loss')

        plt.show()



# In[25]:


if __name__ == '__main__':
    main()
    

