import os
import sys
import argparse
import numpy as np

import torch
import torchvision

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

import ops
import utils
import datagen
from resnet import Generator, Discriminator, MyConvo2d

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--dim', default=64, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--exp', default='1', type=str)
    parser.add_argument('--output_dim', default=4096, type=int)
    parser.add_argument('--dataset', default='celeba', type=str)

    args = parser.parse_args()
    return args

# iterate with attributes
# have to call datagen.load_celeba_50k_attrs(args)
def inf_gen_attrs(data_gen):
    while True:
        for images, targets, img_attrs in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets, img_attrs)

def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)


def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                nn.init.kaiming_uniform_(m.conv.weight)
            else:
                nn.init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            nn.init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train(args):
    
    torch.manual_seed(8734)
    
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    print (netG, netD)
    netG.apply(weights_init)
    netD.apply(weights_init)
    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    
    celeba_train = datagen.load_celeba_50k(args)
    train = inf_gen(celeba_train)
    print ('saving reals')
    reals, _ = next(train)
    utils.create_if_empty('results') 
    utils.create_if_empty('results/celeba_resnet') 
    utils.create_if_empty('saved_models') 
    utils.create_if_empty('saved_models/celeba_resnet') 
    save_image(reals, 'results/celeba/reals.png') 

    one = torch.tensor(1.).cuda()
    mone = one * -1
    total_batches = 0
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        total_batches += 1
        ops.batch_zero_grad([netG, netD])
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(args.disc_iters):
            data, targets = next(train)
            netD.zero_grad()
            d_real = netD(data).mean()
            d_real.backward(mone, retain_graph=True)
            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            with torch.no_grad():
                fake = netG(noise)
            fake.requires_grad_(True)
            d_fake = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = ops.grad_penalty_3dim(args, netD, data, fake)
            gp.backward()
            d_cost = d_fake - d_real + gp
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        fake = netG(noise)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 5 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
            print('')
        if iter % 50 == 0:
            val_d_costs = []
            #path = 'results/celeba_resnet/iter_{}.png'.format(iter)
            #utils.generate_image(args, netG, path)
        if iter % 5000 == 0:
            print('')
            #utils.save_model('saved_models/celeba_resnet/netG_{}'.format(iter), netG, optimG)
            #utils.save_model('saved_models/celeba_resnet/netD_{}'.format(iter), netD, optimD)
          

if __name__ == '__main__':

    args = load_args()
    train(args)
