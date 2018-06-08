import sys
import argparse
import pickle

import torch as t
import torchvision as tv
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid 

import numpy as np
import pandas as pd
from PIL import Image 
from keras.utils.np_utils import to_categorical


def save_any(anything, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(anything, handle)
        
def load_any(filename):
    with open(filename, 'rb') as handle:
        anything = pickle.load(handle)
    return anything


parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", default=0, type=int, help="thread numbers for processing the dataloader")
parser.add_argument("--image_size", default=64, type=int, help="image size")
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--max_epoch", default=50, type=int, help="max epoch")
parser.add_argument("--lr1", default=0.0002, type=float, help="learning rate for generator")
parser.add_argument("--lr2", default=0.0002, type=float, help="learning rate for discriminator")
parser.add_argument("--beta1", default=0.5, type=float, help="adam parameter beta")
parser.add_argument("--gpu", default=None, help="whether use GPU")
parser.add_argument("--nz", default=100, type=int, help="noise dimension")
parser.add_argument("--ngf", default=64, type=int, help="feature map of generator")
parser.add_argument("--ndf", default=64, type=int, help="feature map of discriminator")
parser.add_argument("--vis", default=True, help="whether use visdom to visualize")
parser.add_argument("--env", default="GAN", help="env of visdom")
parser.add_argument("--plot_every", default=50, type=int, help="visdom plot once every certain fixed batches")
parser.add_argument("--d_every", default=1, type=int, help="train discriminator once every one batch")
parser.add_argument("--g_every", default=5, type=int, help="train generator once every five batches")
parser.add_argument("--save_every", default=1, type=int, help="save model once every ten batches")
parser.add_argument("--gen_img", default="result.png", help="name for generating imgs")
parser.add_argument("--gen_num", default=64, type=int, help="store 64 best imgs among 512 generated imgs")
parser.add_argument("--gen_search_num", default=512, type=int, help="generate 512 imgs")
parser.add_argument("--gen_mean", default=0, type=int, help="mean of noise for generator")
parser.add_argument("--gen_std", default=1, type=int, help="standard deviation of noise for discriminator")
parser.add_argument("--random_seed", default="random_seed.pickle", help="fix random seed for generate function")
args = parser.parse_args(args=[])

hair_conv = load_any('hair_conv')
eyes_conv = load_any('eyes_conv')


def textpairs_condition(text_pairs):
    hair = to_categorical(text_pairs[:, 0].contiguous(), num_classes=12)
    eyes = to_categorical(text_pairs[:, 1].contiguous(), num_classes=10)
    condition = np.concatenate([hair, eyes], axis=1)
    return t.Tensor(condition)


def evaluation(netg, text_pairs, baseline=False, seed=None):
    # text_pairs: ndarray, (batch, 2), where 2 is for hair and eyes
    batch_size = len(text_pairs)
    
    noise = t.randn(batch_size, args.nz, 1, 1)
    if seed is not None:
        t.manual_seed(seed)
        noise = t.randn(batch_size, args.nz, 1, 1)
        
    noise = Variable(noise.cuda(), volatile=True)
    condition = Variable(textpairs_condition(t.LongTensor(text_pairs)).cuda(), volatile=True)
                  
    fake_imgs = netg(noise, condition)
    
    if not baseline:
        showt(fake_imgs.data, size=(8, 8))
    else:
        import matplotlib.pyplot as plt
        r, c = 5, 5
        gen_imgs = fake_imgs[:25].data.cpu().numpy()
        gen_imgs = (((gen_imgs + 1) / 2) * 255).astype(np.uint8)
        gen_imgs = np.transpose(gen_imgs, axes=(0, 2, 3, 1))
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('samples/cgan.png')


class NetG(nn.Module):
    def __init__(self, args):
        super(NetG, self).__init__()
        ngf = args.ngf
        self.condition_dim = 22
        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.nz + self.condition_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        condition = condition.view(condition.size()[0], -1, 1, 1)
        conditioned_noise = t.cat([noise, condition], dim=1)
        return self.main(conditioned_noise)


class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()
        ndf = args.ndf
        self.outdim = 22
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, self.outdim, 4, 1, 0, bias=False),
        )

    def forward(self, image, condition):
        encoded = self.main(image).view(len(image), -1)
        similarity = t.bmm(encoded.unsqueeze(1), condition.unsqueeze(2)).squeeze()
        return F.sigmoid(similarity)


netg = NetG(args).cuda()
netd = NetD(args).cuda()

checkpoint = t.load('cgan_model')
netg.load_state_dict(checkpoint['netg'])
netd.load_state_dict(checkpoint['netd'])


with open(sys.argv[1]) as handle:
    lines = handle.read().split('\n')[:-1]
    assigned = [[line.split()[0].split(',')[1], line.split()[2]] for line in lines]


assign = [(hair_conv[hair], eyes_conv[eyes]) for hair, eyes in assigned]
evaluation(netg, np.array(assign), baseline=True, seed=2)
