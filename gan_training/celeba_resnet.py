import os
import sys
import argparse
import numpy as np
import pandas as pd
import math

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
import bayes_net

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
    netD = Discriminator(args).cuda()
    graph = bayes_net.create_bayes_net()
    #print (netG, netD)
    print('Initializing weights...\n')
    netG.apply(weights_init)
    netD.apply(weights_init)
    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    
    bce_loss = nn.BCEWithLogitsLoss()

    celeba_train = datagen.load_celeba_50k_train(args)
    train = inf_gen_attrs(celeba_train)
    print ('saving reals')
    reals, _, _ = next(train)
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
            data, targets, targ_attrs = next(train)
            netD.zero_grad()
            #print(data.shape)
            true_false, preds = netD(data)
            d_real = true_false.mean()
            d_attrs = bce_loss(preds, targ_attrs.squeeze().float().cuda())
            d_total_loss = d_real - d_attrs
            d_total_loss.backward(mone, retain_graph=True)

            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            marginals = torch.tensor(get_marginals(graph, args.batch_size))
            mdist = torch.distributions.Bernoulli(marginals)
            attr_samples = mdist.sample().cuda().requires_grad_()
            noise = torch.cat((noise, attr_samples), 1)
            with torch.no_grad():
                fake = netG(noise)
            fake.requires_grad_(True)
            d_fake, _ = netD(fake)
            d_fake = d_fake.mean()
            d_fake.backward(one, retain_graph=True)
            gp = ops.grad_penalty_3dim(args, netD, data, fake)
            gp.backward()
            d_cost = d_fake - d_total_loss + gp
            wasserstein_d = d_real - d_fake
            optimD.step()

        for p in netD.parameters():
            p.requires_grad=False
        netG.zero_grad()
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        marginals = torch.tensor(get_marginals(graph, args.batch_size))
        mdist = torch.distributions.Bernoulli(marginals)
        attr_samples = mdist.sample().cuda().requires_grad_()
        noise = torch.cat((noise, attr_samples), 1)
        fake = netG(noise)
        G, fake_preds = netD(fake)
        G = G.mean()
        g_attrs = bce_loss(fake_preds, attr_samples)
        classif_weight = 12
        classif_coef = classif_weight*math.exp(-(2500/(iter+1)))
        G = G - classif_coef*g_attrs

        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 10 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
            print('iter: ', iter, 'train D attr', d_attrs.cpu().item())
            print('iter: ', iter, 'train G attr', g_attrs.cpu().item())
            print('')
        if iter % 500 == 0:
            val_d_costs = []
            path = 'results/celeba_resnet/iter_{}.png'.format(iter)
            utils.generate_image(args, attr_samples, netG, path)
        if iter % 5000 == 0:
            print('')
            utils.save_model('saved_models/celeba_resnet/netG_{}'.format(iter), netG, optimG)
            utils.save_model('saved_models/celeba_resnet/netD_{}'.format(iter), netD, optimD)
          

if __name__ == '__main__':

    args = load_args()
    train(args)
