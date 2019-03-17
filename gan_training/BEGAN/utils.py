import os
import sys
import time
import torch
import glob
import itertools
import numpy as np
import pandas as pd
from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
import torch.distributions.uniform as U
from torchvision.utils import save_image

import bayes_net as bn

def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_normal(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def create_uniform(min, max):
    min = torch.tensor([min]).float()
    max = torch.tensor([max]).float()
    D = U.Uniform(min, max)
    return D


def sample_z(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape)).cuda()
    z.requires_grad = grad
    return z


def sample_z_like(shape, scale=1., grad=True):
    return torch.randn(*shape, requires_grad=grad).cuda()


def create_if_empty(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(path, model, optim):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        }, path)


def load_model(args, model, optim, path):
    path = model_dir + path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    return model, optim


def get_net_only(model):
    net_dict = {'state_dict': model.state_dict()}
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model


def get_marginals(graph, batch_size):
    df = pd.DataFrame(columns=bn.KEEP_ATTS)
    # only need to generate one set of random evidence per batch
    evidence = bn.random_evidence()
    targets = []
    for val in bn.KEEP_ATTS:
        if val not in evidence.keys():
            targets.append(val)
    query = bn.graph_inference(graph, targets, evidence)

    # just repeat it for the entire batch, since we want to pass the same evidence in
    for i in range(batch_size):
        for val in bn.KEEP_ATTS:
            if val not in targets:
                df.loc[i, val] = 1
            else:
                df.loc[i, val] = query[val].values[1]
                #print(val, query[val].values[1])

    df = df.apply(pd.to_numeric, downcast='float', errors='coerce')
    return df.values


"""
def load_feature_detector(args):
    print('Loading feature detector...\n')
    fdet = AttributeDetector(args).cuda()
    fdet.load_state_dict(torch.load('saved_models/celeba/netFDET_best_r2'))
    for p in fdet.parameters():
        p.requires_grad = False
    fdet.eval()
    return fdet
"""

def evidence_test(args):
    torch.manual_seed(8734)
    print('Evidence test...\n')
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    fdet = load_feature_detector(args)
    graph = bn.create_bayes_net()
    evidence = bn.evidence_query(['Young', 'Glasses'], [1, 1])
    marginals = bn.return_marginals(graph, args.batch_size, evidence)

    bce_loss = BCEWithLogitsLoss()
    noise = torch.randn(args.batch_size, args.z).cuda()
    if args.dist_exp == 1:
        print('Loading models...\n')
        netG_saved = torch.load('./saved_models/celeba_resnet/netG_current_dist')
        netD_saved = torch.load('./saved_models/celeba_resnet/netD_current_dist')
        netG.load_state_dict(netG_saved['state_dict'])
        netD.load_state_dict(netD_saved['state_dict'])

        with torch.no_grad():
            marginals = torch.tensor(marginals).cuda()
            fake = netG(noise, marginals)
            G, _ = netD(fake)
            G_attrs = torch.sigmoid(fdet(fake))
            classif_loss = torch.sum(torch.min(G_attrs, 1-G_attrs))
            G_attrs = torch.round(torch.sigmoid(G_attrs)).mean(0)
            dist_loss = mseloss(G_attrs, marginals[0])
            path = 'results/celeba_resnet/evtest_dist.png'
            utils.generate_image(args, marginals, netG, path)

            print('Distribution Loss', dist_loss.cpu().item())
            print('Classification Loss', classif_loss.cpu().item())

    else:
        print('Loading models...\n')
        netG_saved = torch.load('./saved_models/celeba_resnet/netG_current')
        netD_saved = torch.load('./saved_models/celeba_resnet/netD_current')
        netG.load_state_dict(netG_saved['state_dict'])
        netD.load_state_dict(netD_saved['state_dict'])
        optimG.load_state_dict(netG_saved['optimizer'])
        optimD.load_state_dict(netD_saved['optimizer'])

        with torch.no_grad():
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


def generate_image(args, netG, path):
    with torch.no_grad():
        noise = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
        samples = netG(noise)
    if samples.dim() < 4:
        channels = 1
        samples = samples.view(-1, channels, 28, 28)
    else:
        channels = samples.shape[1]
        samples = samples.view(-1, channels, 64, 64)
        samples = samples.mul(0.5).add(0.5) 
    print ('saving sample: ', path)
    save_image(samples, path)
