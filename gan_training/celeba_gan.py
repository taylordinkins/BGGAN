import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torchvision

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

import ops
import utils
import datagen
import bayes_net
import resnet

def load_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--dim', default=128, type=int, help='latent space width')
    parser.add_argument('--l', default=10, type=int, help='latent space width')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--disc_iters', default=5, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--exp', default='1', type=str)
    parser.add_argument('--output', default=4096, type=int)
    parser.add_argument('--dataset', default='celeba', type=str)

    args = parser.parse_args()
    return args

# maybe I'll clean this up later, maybe not
def load_args_fdet():

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


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Generator'
        # add on +9 for the number of attributes
        self.linear1 = nn.Linear(self.z+9, 4*4*4*self.dim)
        self.conv1 = nn.ConvTranspose2d(4*self.dim, 2*self.dim, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(2*self.dim, 2*self.dim, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(2*self.dim, self.dim, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(self.dim, 3, 4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm1d(4*4*4*self.dim)
        self.bn1 = nn.BatchNorm2d(2*self.dim)
        self.bn2 = nn.BatchNorm2d(2*self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print ('G in: ', x.shape)
        x = self.relu(self.bn0(self.linear1(x)))
        x = x.view(-1, 4*self.dim, 4, 4)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.tanh(x)
        x = x.view(-1, 3, 64, 64)
        # print ('G out: ', x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.name = 'Discriminator'
        self.conv1 = nn.Conv2d(3, self.dim, 5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(4*self.dim, 1, 5, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(4*4*4*self.dim, 1)
        self.ln1 = nn.LayerNorm([self.dim, 30, 30])
        self.ln2 = nn.LayerNorm([self.dim*2, 13, 13])
        self.ln3 = nn.LayerNorm([self.dim*4, 5, 5])
        self.d1 = nn.Dropout(.5)
        self.d2 = nn.Dropout(.5)
        self.d3 = nn.Dropout(.5)

        # attr detection stuff
        self.conv5 = nn.Conv2d(self.dim*4, 9, 5, stride=1, padding=0, bias=False)

    def forward(self, x, xn=False, fdetect=False):
        # print ('D in: ', x.shape)
        x = x.view(-1, 3, 64, 64)
        x = self.d1(self.relu(self.ln1(self.conv1(x))))
        x = self.d1(self.relu(self.ln2(self.conv2(x))))
        xx = self.d1(self.relu(self.ln3(self.conv3(x))))
        #x = x.view(-1, 4*4*4*self.dim)
        # print ('D out: ', x.shape)
        if fdetect is False:
            x = self.conv4(xx).view(-1, 1)
            #print(x.shape)
            if xn is False:
                return x
            else:
                print (x.shape, xx.shape)
                return x, xx
        else:
            x = self.conv5(xx).squeeze()
            #print(x.shape)
            return x



# iterate with attributes
# have to call datagen.load_celeba_50k_attrs(args)
def inf_gen_attrs(data_gen):
    while True:
        for images, targets, img_attrs in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            #img_attrs[img_attrs==-1] = 0
            yield (images, targets, img_attrs)


def inf_gen(data_gen):
    while True:
        for images, targets in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            yield (images, targets)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    if classname.find('Linear') != -1:
        if m.weight.data is not None:
            nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def get_marginals(graph, batch_size):
    df = pd.DataFrame(columns=bayes_net.KEEP_ATTS)
    
    # only need to generate one set of random evidence per batch
    evidence = bayes_net.random_evidence()

    targets = []
    for val in bayes_net.KEEP_ATTS:
        if val not in evidence.keys():
            targets.append(val)
    query = bayes_net.graph_inference(graph, targets, evidence)

    # just repeat it for the entire batch, since we want to pass the same evidence in
    for i in range(batch_size):
        for val in bayes_net.KEEP_ATTS:
            if val not in targets:
                df.loc[i, val] = 1
                #print(val, 1)
            else:
                df.loc[i, val] = query[val].values[1]
                #print(val, query[val].values[1])
        
    df = df.apply(pd.to_numeric, downcast='float', errors='coerce')
    return df.values

def train(args):
    
    torch.manual_seed(8734)
    
    netG = Generator(args).cuda()
    #netG = Generator(args)
    netD = Discriminator(args).cuda()
    #netD = Discriminator(args)
    #print (netG, netD)

    graph = bayes_net.create_bayes_net()

    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)

    # mseloss for penalizing the generator distribution versus the evidence
    # call on mean of batch of fakes
    mseloss = nn.MSELoss()
    
    #celeba_train = datagen.load_celeba_50k(args)
    celeba_train = datagen.load_celeba_50k_attrs(args)
    train = inf_gen_attrs(celeba_train)
    print ('saving reals')
    reals, _, _ = next(train)
    utils.create_if_empty('results') 
    utils.create_if_empty('results/celeba') 
    utils.create_if_empty('saved_models') 
    utils.create_if_empty('saved_models/celeba') 
    save_image(reals, 'results/celeba/reals.png') 

    one = torch.tensor(1.).cuda()
    #one = torch.tensor(1.)
    mone = one * -1
    total_batches = 0
    
    print ('==> Begin Training')
    for iter in range(args.epochs):
        total_batches += 1
        ops.batch_zero_grad([netG, netD])
        for p in netD.parameters():
            p.requires_grad = True
        for _ in range(args.disc_iters):
            data, targets, targ_attrs = next(train)
            netD.zero_grad()
            d_real = netD(data).mean()
            d_real.backward(mone, retain_graph=True)

            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            marginals = torch.tensor(get_marginals(graph, args.batch_size), requires_grad=True).cuda()
            noise = torch.cat((noise, marginals), 1)
            #noise = torch.randn(args.batch_size, args.z, requires_grad=True)
            with torch.no_grad():
                fake = netG(noise)
                print(fake.shape)
            fake.requires_grad_(True)
            d_fake = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = ops.grad_penalty_3dim(args, netD, data, fake)
            #ct = ops.consistency_term(args, netD, data)
            # set it to 0 for now, avoid tensor problems
            ct = 0
            gp.backward()
            d_cost = d_fake - d_real + gp + (2 * ct)
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        marginals = torch.tensor(get_marginals(graph, args.batch_size), requires_grad=True).cuda()
        noise = torch.cat((noise, marginals), 1)
        #noise = torch.randn(args.batch_size, args.z, requires_grad=True)
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
        if iter % 500 == 0:
            val_d_costs = []
            #path = 'results/celeba/iter_{}.png'.format(iter)
            #utils.generate_image(args, netG, path)
        if iter % 5000 == 0:
            print('')
            #utils.save_model('saved_models/celeba/netG_{}'.format(iter), netG, optimG)
            #utils.save_model('saved_models/celeba/netD_{}'.format(iter), netD, optimD)

# train seperately, it was messing up the backprop -_-
def train_feature_detector(args):
    torch.manual_seed(8734)

    # just use the resnet, why not
    fdet = resnet.AttributeDetector(args).cuda()
    optimF = optim.Adam(fdet.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-5)

    criterion = nn.BCEWithLogitsLoss()
    celeba_train = datagen.load_celeba_50k_attrs(args)
    train = inf_gen_attrs(celeba_train)

    total_batches = 0

    print ('==> Begin Training')
    for iter in range(args.epochs):
        total_batches += 1
        fdet.zero_grad()
        for p in fdet.parameters():
            p.requires_grad = True
        data, targets, targ_attrs = next(train)
        #print(data.shape)
        targ_attrs = targ_attrs.squeeze().float().cuda()
        output = fdet(data)
        loss = criterion(output, targ_attrs)
        loss.backward()
       
        optimF.step()

        if iter % 50 == 0:
            print('iter: ', iter, 'BCE Loss', loss.cpu().item())
            print(torch.sigmoid(output[0]), targ_attrs[0])
            print('')
        if iter % 500 == 0:
            val_d_costs = []
            #path = 'results/celeba/iter_{}.png'.format(iter)
            #utils.generate_image(args, netG, path)
        if iter % 5000 == 0:
            print('')
            utils.save_model('saved_models/celeba/netFDET_{}'.format(iter), fdet, optimF)




if __name__ == '__main__':
    #print("I'm not using cuda, because laptop reasons\n")
    # args = load_args()
    # train(args)

    args = load_args_fdet()
    train_feature_detector(args)
