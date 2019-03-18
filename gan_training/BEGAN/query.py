import os
import os.path
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.distributions import Binomial, kl_divergence

import utils
from models import Generator, Discriminator
from dataloader import *
import bayes_net
import datagen

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', default=64, type=int)
    parser.add_argument('--nc', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_lower_boundary', default=2e-6, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--lambda_k', default=0.001, type=float)
    parser.add_argument('--k', default=0, type=float)
    parser.add_argument('--scale', default=64, type=int)
    parser.add_argument('--name', default='c-began-query-experiment')
    parser.add_argument('--base_path', default='./')
    parser.add_argument('--data_path', default='/home/npmhung/workspace/coursework/cs536/BGGAN/data/img_align_celeba')
    parser.add_argument('--load_step', default=0, type=int)
    parser.add_argument('--print_step', default=100, type=int)
    parser.add_argument('--resume_G', default=None)
    parser.add_argument('--resume_D', default=None)
    args = parser.parse_args()
    return args


def prepare_paths(args):
    utils.create_if_empty('./experiments')
    utils.create_if_empty('./experiments/{}/'.format(args.name))
    utils.create_if_empty('./experiments/{}/models'.format(args.name))
    utils.create_if_empty('./experiments/{}/samples'.format(args.name))
    utils.create_if_empty('./experiments/{}/params'.format(args.name))
    sample_path = './experiments/{}/samples'.format(args.name)
    print("Generated samples saved in {}".format(sample_path))
    return


def init_models(args):
    netG = Generator(args).cuda()
    netD = Discriminator(args).cuda()
    print (netG, netD)

    optimG = torch.optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=args.lr)
    optimD = torch.optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=args.lr)

    str = 'experiments/{}/models/'.format(args.name)
    if args.resume_G is not None:
        netG, optimG = utils.load_model(netG, optimG, str+args.resume_G)
    if args.resume_D is not None:
        netD, optimD = utils.load_model(netG, optimD, str+args.resume_D)
    return (netG, optimG), (netD, optimD)


def save_images(args, sample, recon, step, nrow=8):
    save_path = 'experiments/{}/{}_gen.png'.format(args.name, step)
    save_image(sample, save_path, nrow=nrow, normalize=True)
    if recon is not None:
        save_path = 'experiments/{}/{}_disc.png'.format(args.name, step)
        save_image(recon, save_path, nrow=nrow, normalize=True)
    return

def load_part_model(model, path):
    modeldict = model.state_dict()
    pretrained = torch.load(path)['state_dict']
    pretrained = {k:v for k, v in pretrained.items() if k in modeldict}
    modeldict.update(pretrained)
    model.load_state_dict(modeldict)
    return model

def query(args, evidence={}):
    prepare_paths(args)
    (netG, optimG), (netD, optimD) = init_models(args)
    load_part_model(netG, './experiments/c-began/models/netG_20000.pt')
    graph = bayes_net.create_bayes_net()
    z = torch.FloatTensor(args.batch_size, args.z).cuda()
    z.data.uniform_(-1, 1).view(args.batch_size, args.z)
    marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence)).cuda()
    
    
    with torch.no_grad():
        g_fake, g_feats = netG(z, marginals)
    print(marginals[0])
    print(g_feats.mean(dim=0))
    print((g_feats>0.5).float().mean(dim=0))
    save_images(args, g_fake, None, 0)
    
if __name__ == "__main__":
    args = load_args()
    query(args, {'Wearing_Lipstick':1})