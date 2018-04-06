
# coding: utf-8

# In[2]:


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

def parameter_line(model1, model2, alpha = 0.0):
    model = LeNet()
    model1_para = list(model1.parameters())
    model2_para = list(model2.parameters())
    for i, par in enumerate(model.parameters()):
        par.data = alpha * model2_para[i].data + (1.0-alpha) * model1_para[i].data
    model.cuda()
    return model

def count_surface(model1, model2, loader, alpha = (-1.0,2.0),):
    
    inte = 0.1
    alpha_l, alpha_r = alpha
    alphas = np.arange(alpha_l, alpha_r, inte)
    
    total_point = (alpha_r-alpha_l)/inte
    surface = []
    accs = []
    for j,al in enumerate(alphas):
        flag = time.time()
        epoch_loss = 0.0
        t_acc = 0.0
        model = parameter_line(model1, model2, alpha = al)
        for i, data in enumerate(loader, 0):
            inputs, y = data
            inputs, y = Variable(inputs.cuda()), Variable(y.cuda())

            outputs = model(inputs)
            
            temp_loss = criterion(outputs, y)
            epoch_loss += temp_loss.data[0]
            
            pred = torch.max(outputs.data, 1)[1]
            d = pred.eq(y.data).cpu()
            accuracy = d.sum()/d.size()[0]
            t_acc += accuracy
        surface.append(epoch_loss/(i+1))
        accs.append(t_acc/(i+1))
        print("finished %.3f 's iteration" % (j/(total_point+0.0)))
    return np.array(surface) ,np.array(accs) , np.array(alphas)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data_mnist', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data_mnist', train=False,
                                       download=True, transform=transform)


torch.manual_seed(100)
gain = 30
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        torch.manual_seed(1)
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)
        self.fc1 = nn.Linear(4*4*20, 100)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)
        self.fc2 = nn.Linear(100, 10)
        init.xavier_uniform(self.conv1.weight.data, gain = gain)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"

model1 = LeNet()
model2 = LeNet()
ttt = LeNet()
init_net = LeNet()
init_net2 = LeNet()
model1.cuda()
model2.cuda()
ttt.cuda()

torch.manual_seed(100)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=2)

#trainloader_t = torch.utils.data.DataLoader(trainset, batch_size=1000,shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)

batch_str1 = '512'
batch_str2 = '8'
text = 'model/mnist_lenet_batchsize_'
cpu = '_cpu'
model1.load_state_dict(torch.load(text + batch_str1 ))
model2.load_state_dict(torch.load(text + batch_str2 ))


# # draw the surface
# 

# In[ ]:


surface, accs, alphas = count_surface(model1, model2, trainloader, alpha = (-1.0, 2.0))
surface_t, accs_t, alphas = count_surface(model1, model2, testloader, alpha = (-1.0, 2.0))

cut = 0
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
#par2 = host.twinx()

host.plot(alphas[cut:], np.log(surface[cut:]), color = 'blue', label = 'train')
host.plot(alphas[cut:],np.log(surface_t[cut:]),linestyle='dashed',color = 'blue', label = 'test')
host.set_ylabel("cross_entropy", color= 'blue')

par1.plot(alphas[cut:], accs[cut:],color = 'red')
par1.plot(alphas[cut:],accs_t[cut:],linestyle='dashed',color = 'red')
par1.set_ylabel("accuracy", color= 'red')
host.set_xlabel('alpha')
host.legend()

cut = 0
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
#par2 = host.twinx()

host.plot(alphas[cut:], np.log(surface[cut:]), color = 'blue', label = 'train')
host.plot(alphas[cut:],np.log(surface_t[cut:]),linestyle='dashed',color = 'blue', label = 'test')
host.set_ylabel("cross_entropy", color= 'blue')

par1.plot(alphas[cut:], accs[cut:],color = 'red')
par1.plot(alphas[cut:],accs_t[cut:],linestyle='dashed',color = 'red')
par1.set_ylabel("accuracy", color= 'red')
host.set_xlabel('alpha')
host.legend()


# # count the sensitivity and the sensitivity surface

# In[10]:


batches = [4,8,16,32,64,128,256,512,1024,2048,4096,10000]
batches = [32,64,128]
stepes = [0.001,0.01,0.1,1.0,5.0,10.0]
for batch in batches:
    batch_str1 = str(batch)
    text = 'mnist_lenet_batchsize_'

    for step in stepes:
        sens = []
        #step = 10.0
        step_sizes = np.arange(-10*step,10 * step, step)
        sensitivity = 0.0
        run_loss = 0.0
        model1.load_state_dict(torch.load(text + str(batch_str1) ))

        for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda(), requires_grad = True), Variable(labels.cuda())

                line_inputs = torch.ones(inputs.size())
                line_inputs = Variable(line_inputs.cuda())

                outputs = model1(inputs)
                loss = criterion(outputs, labels)
                lower_loss = loss.data[0]

                grad_params = torch.autograd.grad(loss, inputs, create_graph=True,)
                #grad_param = torch.FloatTensor(grad_params)
                #set_trace()
                direction = grad_params[0]
                direction = F.normalize(direction, p = 2, dim = 0)

                for step_size in step_sizes:
                    line_inputs.data = inputs.data - direction.data * step_size
                    line_outputs = model1(line_inputs)
                    loss = criterion(line_outputs, labels)
                    upper_loss = loss.data[0]

                    sensitivity = upper_loss - lower_loss
                    sensitivity = upper_loss 

                    sens.append(sensitivity)
                    #print(sensitivity)
        np.save( 'sens_test_' + str(step) + '_' + batch_str1,sens)

batches = [4,8,16,32,64,128,256,512,1024,2048,4096,10000]
batches = [32,64,128]
stepes = [0.001,0.01,0.1,1.0,5.0,10.0]
for batch in batches:
    batch_str1 = str(batch)
    text = 'mnist_lenet_batchsize_'

    for step in stepes:
        sens = []
        #step = 10.0
        step_sizes = np.arange(-10*step,10 * step, step)
        sensitivity = 0.0
        run_loss = 0.0
        model1.load_state_dict(torch.load(text + str(batch_str1) ))

        for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda(), requires_grad = True), Variable(labels.cuda())

                line_inputs = torch.ones(inputs.size())
                line_inputs = Variable(line_inputs.cuda())

                outputs = model1(inputs)
                loss = criterion(outputs, labels)
                lower_loss = loss.data[0]

                grad_params = torch.autograd.grad(loss, inputs, create_graph=True,)
                #grad_param = torch.FloatTensor(grad_params)
                #set_trace()
                direction = grad_params[0]
                direction = F.normalize(direction, p = 2, dim = 0)

                for step_size in step_sizes:
                    line_inputs.data = inputs.data - direction.data * step_size
                    line_outputs = model1(line_inputs)
                    loss = criterion(line_outputs, labels)
                    upper_loss = loss.data[0]

                    sensitivity = upper_loss - lower_loss
                    sensitivity = upper_loss 

                    sens.append(sensitivity)
                    #print(sensitivity)
        np.save( 'sens_train_' + str(step) + '_' + batch_str1,sens)


# In[193]:


batches = [4,8,16,32 ,512, 1024, 2048, 4096, 10000]
step =10.0
for batch in batches:
    step_sizes = np.arange(-10*step,10 * step, step)
    sen = 'sens_test_' + str(step) + '_' + str(batch) + '.npy'
    plt.plot(step_sizes,np.load(sen), label = str(batch))
    
plt.xlabel('step_size')
plt.ylabel('loss')
plt.legend()


# In[140]:


batches = [4,8,16,32, 128, 512, 1024, 2048, 4096, 10000]
#batches = [4,8,16,32,  512, 1024, 2048, 4096, 10000]

step = 5.0
sens_t = []
for batch in batches:
    step_sizes = np.arange(-10*step,10 * step, step)
    sen = 'sens_test_' + str(step) + '_' + str(batch) + '.npy'
    temp = np.load(sen)
    #alpha = cos_alpha(step_sizes[0]/50, temp[0], step_sizes[10]/50, temp[10], step_sizes[19]/50, temp[19])
    #alpha = math.acos(alpha)
    alpha = temp[5] - temp[10]
    sens_t.append(alpha)


# In[169]:


fig, host = plt.subplots()
par1 = host.twinx()

#host.semilogx(batches, loss_test, color = 'blue', label = 'test')
#host.semilogx(batches,loss_train,linestyle='dashed',color = 'blue', label = 'train')
#host.set_ylabel("loss", color= 'blue')

par1.semilogx(batches, sens_t,color = 'red')
par1.set_ylabel("sharpness", color= 'red')

host.set_xlabel('epoch')
host.legend()

