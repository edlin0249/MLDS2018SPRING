import os
import gc
import sys
import time
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
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


def now():
    return str(time.time()).split('.')[0]

def save_any(anything, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(anything, handle)
        
def load_any(filename):
    with open(filename, 'rb') as handle:
        anything = pickle.load(handle)
    return anything

def save_checkpoint(items, names, filename):
    state = {}
    for item, name in zip(items, names):
        state[name] = item.state_dict()
    
    t.save(state, filename)


version_id = 'cgan_tmp'
parser = argparse.ArgumentParser()
parser.add_argument("--action", default="train", help="generate or train")
parser.add_argument("--data_path", default=None, help="data storage path")
parser.add_argument("--text_path", default=None, help="data storage path")
parser.add_argument("--image_path", default=os.path.join(version_id, "images"), help="storage path of pictures of generator")
parser.add_argument("--model_path", default=os.path.join(version_id, "checkpoints"), help="path for storing NetD and NetG")
parser.add_argument("--saved_model", default=None, help="path for storing NetD")
parser.add_argument("--num_workers", default=0, type=int, help="thread numbers for processing the dataloader")
parser.add_argument("--image_size", default=64, type=int, help="image size")
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--max_epoch", default=100, type=int, help="max epoch")
parser.add_argument("--lr1", default=0.0002, type=float, help="learning rate for generator")
parser.add_argument("--lr2", default=0.0002, type=float, help="learning rate for discriminator")
parser.add_argument("--beta1", default=0.5, type=float, help="adam parameter beta")
parser.add_argument("--gpu", default=None, help="whether use GPU")
parser.add_argument("--nz", default=100, type=int, help="noise dimension")
parser.add_argument("--ngf", default=64, type=int, help="feature map of generator")
parser.add_argument("--ndf", default=64, type=int, help="feature map of discriminator")
parser.add_argument("--vis", default=True, help="whether use visdom to visualize")
parser.add_argument("--env", default="GAN", help="env of visdom")
parser.add_argument("--d_every", default=1, type=int, help="train discriminator once every one batch")
parser.add_argument("--g_every", default=5, type=int, help="train generator once every five batches")
parser.add_argument("--save_every", default=1, type=int, help="save model once every ten batches")
parser.add_argument("--gen_img", default="result.png", help="name for generating imgs")
parser.add_argument("--gen_num", default=64, type=int, help="store 64 best imgs among 512 generated imgs")
parser.add_argument("--gen_search_num", default=512, type=int, help="generate 512 imgs")
parser.add_argument("--gen_mean", default=0, type=int, help="mean of noise for generator")
parser.add_argument("--gen_std", default=1, type=int, help="standard deviation of noise for discriminator")
parser.add_argument("--random_seed", default="random_seed.pickle", help="fix random seed for generate function")
args = parser.parse_args()

if os.path.exists(version_id) == False:
    os.mkdir(version_id)
if os.path.exists(args.image_path) == False:
    os.mkdir(args.image_path)
if os.path.exists(args.model_path) == False:
    os.mkdir(args.model_path)


def fit_transform_text(filepath):
    text_lines = pd.read_csv(filepath, header=None).values[:, 1]
    text_pairs = [(line.split()[0], line.split()[2]) for line in text_lines]
    hair_conv = {}
    eyes_conv = {}
    for hair, eyes in text_pairs:
        if hair not in hair_conv:
            index = int(len(hair_conv) / 2)
            hair_conv[hair] = index
            hair_conv[index] = hair
        if eyes not in eyes_conv:
            index = int(len(eyes_conv) / 2)
            eyes_conv[eyes] = index
            eyes_conv[index] = eyes
    
    text_pairs = np.array([[hair_conv[hair], eyes_conv[eyes]] for hair, eyes in text_pairs])
    return text_pairs, hair_conv, eyes_conv


class ImageDataSet(data.Dataset):
    def __init__(self, args, transforms=None):
        imgs = os.listdir(args.data_path)
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))
        self.imgs = [os.path.join(args.data_path, img) for img in imgs]
        self.transforms = transforms
        
        info = fit_transform_text(args.text_path)
        self.text_pairs = info[0]
        self.hair_conv = info[1]
        self.eyes_conv = info[2]
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, self.text_pairs[index], index

    def __len__(self):
        return len(self.imgs)


transforms = tv.transforms.Compose([
        tv.transforms.Resize(args.image_size),
        tv.transforms.CenterCrop(args.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


print('Loading training images')
dataset = ImageDataSet(args, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=True, num_workers=args.num_workers, drop_last=False)

save_any(dataset.hair_conv, os.path.join(version_id, 'hair_conv'))
save_any(dataset.eyes_conv, os.path.join(version_id, 'eyes_conv'))


def textpairs_condition(text_pairs):
    hair = to_categorical(text_pairs[:, 0].contiguous(), num_classes=12)
    eyes = to_categorical(text_pairs[:, 1].contiguous(), num_classes=10)
    condition = np.concatenate([hair, eyes], axis=1)
    return t.Tensor(condition)


batch_num = len(list(iter(dataloader)))
def train(args, dataloader, model, opt):
    netd, netg = model
    optimizer_d, optimizer_g = opt
    
    criterion = t.nn.BCELoss()
    fix_noises = Variable(t.randn(args.batch_size, args.nz, 1, 1).cuda(), volatile=True)
    fix_condition = Variable(textpairs_condition(t.LongTensor([[dataset.hair_conv['blonde'], 
                                                                dataset.eyes_conv['yellow']]] * args.batch_size)).cuda(),
                                                                volatile=True)
    
    for epoch in iter(range(args.max_epoch)):
        print('EPOCH ' + str(epoch))
        for ii, (imgs, text_pairs, img_id) in enumerate(dataloader):
            if ii % 50 == 0:
                print(str(ii / batch_num).split('.')[1][:2] + '% finished')
            batch_size = imgs.shape[0]
            real_imgs = Variable(imgs.cuda())
            condition = Variable(textpairs_condition(text_pairs).cuda())
            
            true_labels = Variable(t.ones(batch_size).cuda())
            fake_labels = Variable(t.zeros(batch_size).cuda())            
            
            if ii % args.d_every == 0:
                optimizer_d.zero_grad()
                
                critic_real = netd(real_imgs, condition)
                err_d_real = criterion(critic_real, true_labels)
                
                random_text_pairs = np.concatenate([np.random.randint(12, size=batch_size).reshape(-1, 1), 
                                                    np.random.randint(10, size=batch_size).reshape(-1, 1)], axis=1)
                random_condition = Variable(textpairs_condition(t.LongTensor(random_text_pairs)).cuda())
                critic_mismatch = netd(real_imgs, random_condition)
                err_d_mismatch = criterion(critic_mismatch, fake_labels)
                
                noises = Variable(t.randn(batch_size, args.nz, 1, 1).cuda())
                fake_imgs = netg(noises, condition).detach()
                critic_fake = netd(fake_imgs, condition)
                err_d_fake = criterion(critic_fake, fake_labels)
                
                loss = err_d_real + err_d_fake + err_d_mismatch
                loss.backward()
                optimizer_d.step()

            if ii % args.g_every == 0:
                optimizer_g.zero_grad()
                
                noises = Variable(t.randn(batch_size, args.nz, 1, 1).cuda())
                fake_imgs = netg(noises, condition)
                output = netd(fake_imgs, condition)
                err_g = criterion(output, true_labels)
                
                err_g.backward()
                optimizer_g.step()
    
        if epoch % args.save_every == 0:
            fix_fake_imgs = netg(fix_noises, fix_condition)
            imgname = os.path.join(args.image_path, now() + '.png')
            tv.utils.save_image(fix_fake_imgs.data[:64], imgname, normalize=True, range=(-1, 1))
            
            modelname = os.path.join(args.model_path, now() + '.md')
            save_checkpoint([netd, netg, optimizer_d, optimizer_g], ['netd', 'netg', 'optimizer_d', 'optimizer_g'], modelname)


class NetG(nn.Module):
    def __init__(self, args):
        super(NetG, self).__init__()
        ngf = args.ngf  # 生成器feature map数
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


print('Defining model')
netg = NetG(args).cuda()
netd = NetD(args).cuda()
optimizer_g = t.optim.Adam(netg.parameters(), args.lr1, betas=(args.beta1, 0.999))
optimizer_d = t.optim.Adam(netd.parameters(), args.lr2, betas=(args.beta1, 0.999))

if args.saved_model is not None:
    checkpoint = t.load(args.saved_model)
    netg.load_state_dict(checkpoint['netg'])
    netd.load_state_dict(checkpoint['netd'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d'])

print('Start training')
train(args, dataloader, [netd, netg], [optimizer_d, optimizer_g])


print('Models saved at cgan_tmp/checkpoints')