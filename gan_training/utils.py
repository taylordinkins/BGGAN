import os
import sys
import time
import torch
import glob
import itertools
import numpy as np

from scipy.misc import imsave
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
from torchvision.utils import save_image


def sample_z(args, grad=True):
    z = torch.randn(args.batch_size, args.dim, requires_grad=grad).cuda()
    return z


def create_d(shape):
    mean = torch.zeros(shape)
    cov = torch.eye(shape)
    D = N.MultivariateNormal(mean, cov)
    return D


def sample_d(D, shape, scale=1., grad=True):
    z = scale * D.sample((shape,)).cuda()
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


def load_model(args, model, optim):
    path = '{}/{}/{}_{}.pt'.format(
            args.dataset, args.model, model.name, args.exp)
    path = model_dir + path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    acc = ckpt['best_acc']
    loss = ckpt['best_loss']
    return model, optim, (acc, loss)


def get_net_only(model):
    net_dict = {'state_dict': model.state_dict()}
    return net_dict


def load_net_only(model, d):
    model.load_state_dict(d['state_dict'])
    return model


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

