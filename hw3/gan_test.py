import os
import torch as t 
import torchvision as tv 
from gan_model import NetG, NetD
import argparse
from torch.utils import data
from PIL import Image 
import numpy as np 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle
from pdb import set_trace


class ImageDataSet(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data 

    def __len__(self):
        return len(self.imgs)


def generate(args):
    netg, netd = NetG(args).eval(), NetD(args).eval()
    t.manual_seed(6)
    noises = t.randn(args.gen_search_num, args.nz, 1, 1).normal_(args.gen_mean, args.gen_std)
    noises = Variable(noises, volatile=True)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(args.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(args.netg_path, map_location=map_location))
    
    if args.gpu:
        netd = netd.cuda()
        netg = netg.cuda()
        noises = noises.cuda()

    fake_imgs = netg(noises)
    scores = netd(fake_imgs).data

    indexs = scores.topk(args.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_imgs.data[ii])
    imgs = t.stack(result)
    
    import matplotlib.pyplot as plt
    r, c = 5, 5
    gen_imgs = imgs[:25].cpu().numpy()
    gen_imgs = (((gen_imgs + 1) / 2) * 255).astype(np.uint8)
    gen_imgs = np.transpose(gen_imgs, axes=(0, 2, 3, 1))
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig('samples/gan.png')


def main(args):
    if args.action == "train":
        train(args)
    else:
        generate(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("animation face generation")
    parser.add_argument("action", choices=["train", "generate"], help="generate or train")
    parser.add_argument("--data_path", default="../dataset/faces", help="data storage path")
    parser.add_argument("--num_workers", default=4, type=int, help="thread numbers for processing the dataloader")
    parser.add_argument("--image_size", default=64, type=int, help="image size")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--max_epoch", default=200, type=int, help="max epoch")
    parser.add_argument("-lr1", default=0.0002, type=float, help="learning rate for generator")
    parser.add_argument("-lr2", default=0.0002, type=float, help="learning rate for discriminator")
    parser.add_argument("-beta1", default=0.5, type=float, help="adam parameter beta")
    parser.add_argument("-gpu", default=None, help="whether use GPU")
    parser.add_argument("-nz", default=100, type=int, help="noise dimension")
    parser.add_argument("-ngf", default=64, type=int, help="feature map of generator")
    parser.add_argument("-ndf", default=64, type=int, help="feature map of discriminator")
    parser.add_argument("--save_path", default="images/", help="storage path of pictures of generator")
    parser.add_argument("-vis", default=False, help="whether use visdom to visualize")
    parser.add_argument("-env", default="GAN", help="env of visdom")
    parser.add_argument("--plot_every", default=20, type=int, help="visdom plot once every certain fixed batches")
    parser.add_argument("--d_every", default=1, type=int, help="train discriminator once every one batch")
    parser.add_argument("--g_every", default=5, type=int, help="train generator once every five batches")
    parser.add_argument("--decay_every", default=1, type=int, help="save model once every ten batches")
    parser.add_argument("--model_dir", default="checkpoints/", help="path for storing NetD and NetG")
    parser.add_argument("--netd_path", default=None, help="path for storing NetD")
    parser.add_argument("--netg_path", default=None, help="path for storing NetG")
    parser.add_argument("--gen_img", default="result.png", help="name for generating imgs")
    parser.add_argument("--gen_num", default=64, type=int, help="store 64 best imgs among 512 generated imgs")
    parser.add_argument("--gen_search_num", default=512, type=int, help="generate 512 imgs")
    parser.add_argument("--gen_mean", default=0, type=int, help="mean of noise for generator")
    parser.add_argument("--gen_std", default=1, type=int, help="standard deviation of noise for discriminator")
    args = parser.parse_args()
    args.gpu = t.cuda.is_available()
    main(args)
