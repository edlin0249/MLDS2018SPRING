import torch
import torch.nn as nn
from torch.autograd import Variable
from Utils.CIFAR10ConvNet import CIFAR10ConvNet
from DataLoader.load_data import load_data_train, load_data_test
import torch.optim as optim
import os
import argparse
import matplotlib.pyplot as plt

PATH = './Models/'

def train(args, trainloader, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))

    loss_min = 100
    keep = 0.0
    early_stopping = 0

    for epoch in range(args.epochs):

        running_loss_train = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.data[0] * args.batch_size
            if i % 5 == 4:
                print('[%d, %5d] train_loss: %f' % (epoch + 1, i + 1, running_loss_train / 500))
                running_loss_train = 0.0
                #calculate valid data loss
                """
                running_loss_valid = 0.0
                total = 0.0
                for j, data in enumerate(validloader, 0):
                    inputs, labels = data
                    inputs, labels = Variable(inputs), Variable(labels)

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    running_loss_valid += loss.data[0] * args.batch_size
                    total += labels.data.size(0)

                print('[%d, %5d] valid_loss: %f' % (epoch + 1, i + 1, running_loss_valid / total))
                if loss_min > (running_loss_valid / total):
                    #print('loss_min = %f, (running_loss / total) = %f' % (loss_min, running_loss / total))
                    #print("loss_min > (running_loss / total)")
                    loss_min = (running_loss_valid / total)
                    keep = 1.0
                else:
                    #print('loss_min = %f, (running_loss / total) = %f' % (loss_min, running_loss / total))
                    #print("loss_min < (running_loss / total)")
                    keep += 1.0
                """
            """
            if keep > 5.0:
                early_stopping = 1
                break
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                #running_loss = 0.0
            """
        """
        if early_stopping:
            break
        """

def test(args, trainloader, testloader, Models_file):
    training_loss = []
    training_acc = []
    testing_loss = []
    testing_acc = []
    parameters = []
    for M_path in Models_file:
        net = torch.load(PATH+M_path)
        print(M_path)
        print(int(M_path.split('_')[-2]))
        parameters.append(int(M_path.split('_')[-2]))
        #criterion = nn.CrossEntropyLoss()
        #calculate training data loss and accuracy
        correct = 0
        total = 0
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.data[0] * args.batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        training_loss.append(running_loss/total)
        training_acc.append(correct/total)

		#calculate testing data loss and accuracy
        correct = 0
        total = 0
        running_loss = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.data[0] * args.batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        testing_loss.append(running_loss/total)
        testing_acc.append(correct/total)

    return [parameters, training_loss, training_acc, testing_loss, testing_acc]      



def main():
    parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Example')
    parser.add_argument('action', default='train', choices=['train','test'], help='select one of actions(train/test)')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size(default=100)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.2, help='learning rate(default=0.2)')
    parser.add_argument('--epochs', type=int, default=10, help='epochs(default=10)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum(default=0.5)')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--valid_size', type=float, default=0.2, help='split valid size and train size from original whole training data(default=0.2[20 percent of original whole training data])')
    parser.add_argument('--is_shuffle', type=bool, default=True, help='whether shuffle original whole training data before splitting(default=True)')
    args = parser.parse_args()

    


    if args.action == 'train':

        trainloader, testloader = load_data_test(args)

        net = CIFAR10ConvNet()
        print(net)
        
        #how many parameters in Network
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        print('total parameters = ', pytorch_total_params)
        pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('total trainable parameters = ', pytorch_total_trainable_params)
        train(args, trainloader, net)
        
        torch.save(net, PATH+"model_CIFAR10ConvNet_"+str(pytorch_total_trainable_params)+"_params")
        print('Finished Training')

    elif args.action == 'test':
        trainloader, testloader = load_data_test(args)

        Models_file = os.listdir(PATH)
        parameters, training_loss, training_acc, testing_loss, testing_acc = test(args, \
        	trainloader, testloader, Models_file)
        train_loss, = plt.plot(parameters, training_loss, 'bo', label='train_loss')
        test_loss, = plt.plot(parameters, testing_loss, 'yo', label='test_loss')
        plt.legend(handles=[train_loss, test_loss], loc='best')
        plt.xlabel('number of parameters')
        plt.ylabel('loss')
        plt.title('model loss')

        plt.show()

        train_acc, = plt.plot(parameters, training_acc, 'bo', label='train_acc')
        test_acc, = plt.plot(parameters, testing_acc, 'yo', label='test_acc')
        plt.legend(handles=[train_acc, test_acc], loc='best')
        plt.xlabel('number of parameters')
        plt.ylabel('accuracy')
        plt.title('model accuracy')

        plt.show()



if __name__ == '__main__':
    main()