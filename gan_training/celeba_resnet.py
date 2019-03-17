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
from resnet import Generator, Discriminator, MyConvo2d, AttributeDetector
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
    parser.add_argument('--load_models', default=0, type=int)
    parser.add_argument('--dist_exp', default=0, type=int)
    parser.add_argument('--evidence_test', default=0, type=int)

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

def evidence_test(args):
    torch.manual_seed(8734)
    print('Evidence test...\n')
    
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    fdet = load_feature_detector(args)
    graph = bayes_net.create_bayes_net()
    evidence = bayes_net.evidence_query(['Young', 'Glasses'], [1, 1])
    marginals = bayes_net.return_marginals(graph, args.batch_size, evidence)

    bce_loss = BCEWithLogitsLoss()
    noise = torch.randn(args.batch_size, args.z).cuda()
    if args.dist_exp == 1:
        print('Loading models...\n')
        netG_saved = torch.load('./saved_models/celeba_resnet/netG_current_dist')
        netD_saved = torch.load('./saved_models/celeba_resnet/netD_current_dist')
        netG.load_state_dict(netG_saved['state_dict'])
        netD.load_state_dict(netD_saved['state_dict'])
        for p in netD.parameters():
            p.requires_grad = False
        for pp in netG.parameters():
            p.requires_grad = False

        marginals = torch.tensor(marginals).cuda()
        fake = netG(noise, marginals)
        G, _ = netD(fake)
        G_attrs = torch.sigmoid(fdet(fake))
        classif_loss = torch.sum(torch.min(G_attrs, 1-G_attrs))
        G_attrs = torch.round(G_attrs).mean(0)
        dist_loss = mseloss(G_attrs, marginals[0])
        path = 'results/celeba_resnet/evtest_dist.png'
        utils.generate_image(args, marginals, netG, path)

        print('Distribution Loss', dist_loss.cpu().item())
        print('Classification Loss', classif_loss.cpu().item())
        print()

        


    else:
        print('Loading models...\n')
        netG_saved = torch.load('./saved_models/celeba_resnet/netG_current')
        netD_saved = torch.load('./saved_models/celeba_resnet/netD_current')
        netG.load_state_dict(netG_saved['state_dict'])
        netD.load_state_dict(netD_saved['state_dict'])
        optimG.load_state_dict(netG_saved['optimizer'])
        optimD.load_state_dict(netD_saved['optimizer'])
        for p in netD.parameters():
            p.requires_grad = False
        for pp in netG.parameters():
            p.requires_grad = False

        marginals = torch.tensor(marginals)
        mdist = torch.distributions.Bernoulli(marginals)
        attr_samples = mdist.sample().cuda()
        fake = netG(noise, attr_samples)
        G, _ = netD(fake)
        fake_preds = torch.sigmoid(fdet(fake))
        G_attrs = bce_loss(fake_preds, attr_samples)
        path = 'results/celeba_resnet/evtest_sampling.png'
        utils.generate_image(args, attr_samples, netG, path)

        print('Attribute Loss', G_attrs.cpu().item())
        print()
        

def load_feature_detector(args):
    print('Loading feature detector...\n')
    fdet = AttributeDetector(args).cuda()
    fdet.load_state_dict(torch.load('saved_models/celeba/netFDET_best_r2'))
    for p in fdet.parameters():
        p.requires_grad = False
    fdet.eval()

    return fdet
    

def train(args, load_models=False):
    
    torch.manual_seed(8734)
    
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    fdet = load_feature_detector(args)

    graph = bayes_net.create_bayes_net()
    #print (netG, netD)
    if load_models is False:
        print('Initializing weights...\n')
        netG.apply(weights_init)
        netD.apply(weights_init)

    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    
    if load_models:
        print('Loading models...\n')
        netG_saved = torch.load('./saved_models/celeba_resnet/netG_current')
        netD_saved = torch.load('./saved_models/celeba_resnet/netD_current')
        netG.load_state_dict(netG_saved['state_dict'])
        netD.load_state_dict(netD_saved['state_dict'])
        optimG.load_state_dict(netG_saved['optimizer'])
        optimD.load_state_dict(netD_saved['optimizer'])

    bce_loss = nn.BCEWithLogitsLoss()

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
            #print(data.shape)
            true_false, _ = netD(data)
            d_real = true_false.mean()
            d_real.backward(mone, retain_graph=True)

            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            marginals = torch.tensor(get_marginals(graph, args.batch_size))
            mdist = torch.distributions.Bernoulli(marginals)
            attr_samples = mdist.sample().cuda().requires_grad_()
            with torch.no_grad():
                fake = netG(noise, attr_samples)
            fake.requires_grad_(True)
            d_fake, _ = netD(fake)
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
        marginals = torch.tensor(get_marginals(graph, args.batch_size))
        mdist = torch.distributions.Bernoulli(marginals)
        attr_samples = mdist.sample().cuda().requires_grad_()
        fake = netG(noise, attr_samples)
        G, _ = netD(fake)
        fake_preds = torch.sigmoid(fdet(fake))
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
            print('iter: ', iter, 'train G attr', g_attrs.cpu().item())
            print('')
        if iter % 500 == 0:
            val_d_costs = []
            path = 'results/celeba_resnet/iter_{}.png'.format(iter)
            utils.generate_image(args, attr_samples, netG, path)
        if iter % 2500 == 0 and iter > 0:
            print('')
            utils.save_model('saved_models/celeba_resnet/netG_current', netG, optimG)
            utils.save_model('saved_models/celeba_resnet/netD_current', netD, optimD)


def train_dist_exp(args, load_models=False):
    
    torch.manual_seed(8734)
    
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    fdet = load_feature_detector(args)
    graph = bayes_net.create_bayes_net()
    #print (netG, netD)
    if load_models is False:
        print('Initializing weights...\n')
        netG.apply(weights_init)
        netD.apply(weights_init)

    optimG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)
    optimD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0, 0.9), weight_decay=1e-4)


    
    if load_models:
        print('Loading models...\n')
        netG_saved = torch.load('./saved_models/celeba_resnet/netG_current_dist')
        netD_saved = torch.load('./saved_models/celeba_resnet/netD_current_dist')
        netG.load_state_dict(netG_saved['state_dict'])
        netD.load_state_dict(netD_saved['state_dict'])
        optimG.load_state_dict(netG_saved['optimizer'])
        optimD.load_state_dict(netD_saved['optimizer'])

    mseloss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

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
            #print(data.shape)
            true_false, _ = netD(data)
            d_real = true_false.mean()
            d_real.backward(mone, retain_graph=True)

            noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            marginals = torch.tensor(get_marginals(graph, args.batch_size), requires_grad=True).cuda()
            with torch.no_grad():
                fake = netG(noise, marginals)
            fake.requires_grad_(True)
            d_fake, _ = netD(fake)
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
        marginals = torch.tensor(get_marginals(graph, args.batch_size), requires_grad=True).cuda()
        fake = netG(noise, marginals)
        G, _ = netD(fake)
        G = G.mean()
        G_attrs = torch.sigmoid(fdet(fake))
        classif_loss = torch.sum(torch.min(G_attrs, 1-G_attrs))
        G_attrs = torch.round(G_attrs).mean(0)
        dist_loss = mseloss(G_attrs, marginals[0])
        classif_coef = math.exp(-(3000/(iter+1)))
        dist_coef = 15*math.exp(-(3000/(iter+1)))
        G = G - dist_coef*dist_loss - classif_coef*classif_loss

        G.backward(mone)
        g_cost = -G
        optimG.step()
       
        if iter % 10 == 0:
            print('iter: ', iter, 'train D cost', d_cost.cpu().item())
            print('iter: ', iter, 'train G cost', g_cost.cpu().item())
            print('iter: ', iter, 'train G distribution', dist_loss.cpu().item())
            print('iter: ', iter, 'classif G cost', classif_loss.cpu().item())
            print('')
        if iter % 500 == 0:
            val_d_costs = []
            path = 'results/celeba_resnet/dist_iter_{}.png'.format(iter)
            utils.generate_image(args, marginals, netG, path)
        if iter % 2500 == 0 and iter > 0:
            print('')
            utils.save_model('saved_models/celeba_resnet/netG_current_dist', netG, optimG)
            utils.save_model('saved_models/celeba_resnet/netD_current_dist', netD, optimD)
          

if __name__ == '__main__':

    args = load_args()
    if args.evidence_test == 0:
        if args.dist_exp == 0:
            if args.load_models == 1:
                train(args, load_models=True)
            else:
                train(args)
        else:
            print('Running distribution experiment...\n')
            if args.load_models == 1:
                train_dist_exp(args, load_models=True)
            else:
                train_dist_exp(args)

    else:
        evidence_test(args)
