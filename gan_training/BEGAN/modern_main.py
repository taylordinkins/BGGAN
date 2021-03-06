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

import utils
from models import Generator, Discriminator
from dataloader import *


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
    parser.add_argument('--name', default='test3')
    parser.add_argument('--base_path', default='./')
    parser.add_argument('--data_path', default='../../../celeba/val/val')
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


def Disc_loss(args, d_real, data, d_fake, g_fake):
    real_loss_d = (d_real - data).abs().mean()
    fake_loss_d = (d_fake - g_fake).abs().mean()
    return (real_loss_d, fake_loss_d)
        

def Gen_loss(args, g_out, g_fake):
    return (g_out - g_fake).abs().mean()


def train(args):
    random.seed(8722)
    torch.manual_seed(4565)
    measure_history = deque([0]*3000, 3000)
    convergence_history = []
    prev_measure = 1
    
    lr = args.lr
    iters = args.load_step
    prepare_paths(args)

    data_loader = get_loader(args.data_path, args.batch_size, args.scale, 0)

    z = torch.FloatTensor(args.batch_size, args.z).cuda()
    fixed_z = torch.FloatTensor(args.batch_size, args.z).cuda()
    fixed_z.data.uniform_(-1, 1)    
    fixed_x = None
    iter = 0
    (netG, optimG), (netD, optimD) = init_models(args)
    for i in range(args.epochs):
        for _, data in enumerate(data_loader):
            if data.size(0) != args.batch_size:
                continue
            print (iter)
            data = data.cuda()
            if fixed_x is None:
                fixed_x = data
            z.data.uniform_(-1, 1).view(args.batch_size, args.z)
            netD.zero_grad()
            netG.zero_grad()
            with torch.no_grad():
                g_fake = netG(z)
            d_fake = netD(g_fake)
            d_real = netD(data)
           
            real_loss_d, fake_loss_d = Disc_loss(args, d_real, data, d_fake, g_fake)
            lossD = real_loss_d - args.k * fake_loss_d
            lossD.backward()
            optimD.step()
        
            g_fake = netG(z)
            g_out = netD(g_fake)
            lossG = Gen_loss(args, g_out, g_fake)
            lossG.backward()
            optimG.step()
            with torch.no_grad():
                lagrangian = (args.gamma*real_loss_d - fake_loss_d)
                args.k += args.lambda_k * lagrangian
                args.k = max(min(1, args.k), 0)
            
                convg_measure = real_loss_d.item() + lagrangian.abs()
                measure_history.append(convg_measure)
                if iters % args.print_step == 0:
                    print ("Iter: {}, Epoch: {}, D loss: {}, G Loss: {}".format(iter, i, 
                        lossD.item(), lossG.item()))
                    save_images(args, g_fake, d_real, iter)
               
                """update training parameters"""
                lr = args.lr * 0.95 ** (iter//3000)
     
                for p in optimG.param_groups + optimD.param_groups:
                    p['lr'] = lr

                if iter % 1000 == 0:
                    pathG = 'experiments/{}/models/netG_{}.pt'.format(args.name, iter)
                    pathD = 'experiments/{}/models/netD_{}.pt'.format(args.name, iter)
                    utils.save_model(pathG, netG, optimG)
                    utils.save_model(pathD, netD, optimD)
            
                iter += 1


def generative_experiments(args):
    netG, _, netD, _ = load_models(args)
    z = []
    for iter in range(10):
        z0 = np.random.uniform(-1, 1, args.z)
        z10 = np.random.uniform(-1, 1, args.z)

        def slerp(val, low, high):
            low_norm = low/np.linalg.norm(low)
            high_norm = high/np.linalg.norm(high)
            omega = np.arccos(np.clip(np.dot(low_norm, high_norm), -1, 1))
            so = np.sin(omega)
            if so == 0:
                return (1.0 - val) * low + val * high # L'Hopital's rule/LERP
            return np.sin((1.0-val) * omega) / so * low + np.sin(val*omega) / so * high 

        z.append(z0)
        for i in range(1, 9):
            z.append(slerp(i*0.1, z0, z10))
        z.append(z10.reshape(1, args.z)) 
    z = [_.reshape(1, args.z) for _ in z]
    z_var = torch.from_numpy(np.concatenate(z, 0)).float().cuda()
    samples = netG(z_var)
    save_images(samples, None, 'gen_1014_slerp_{}'.format(args.load_step), 10)


if __name__ == "__main__":
    args = load_args()
    train(args)
    generative_experiments(args)
