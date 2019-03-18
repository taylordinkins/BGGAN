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
import datagen
import bayes_net as bn
from models import Generator, Discriminator
from preact_resnet import PreActResNet18 as PreActResNet

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
    parser.add_argument('--name', default='test4')
    parser.add_argument('--load_step', default=0, type=int)
    parser.add_argument('--print_step', default=100, type=int)
    parser.add_argument('--resume', default=False)
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
    netL = PreActResNet().cuda()
    print (netG, netD)

    optimG = torch.optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=args.lr)
    optimD = torch.optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=args.lr)
    optimL = torch.optim.Adam(netL.parameters(), betas=(0.5, 0.999), lr=args.lr)

    if args.resume:
        Gpath = 'experiments/{}/models/netG_{}.pt'.format(args.name, args.load_step)
        Dpath = 'experiments/{}/models/netD_{}.pt'.format(args.name, args.load_step)
        Lpath = 'experiments/{}/models/netL_{}.pt'.format(args.name, args.load_step)
        netG, optimG = utils.load_model(args, netG, optimG, Gpath)
        netD, optimD = utils.load_model(args, netD, optimD, Dpath)
        netL, optimL = utils.load_model(args, netL, optimL, Lpath)
    return (netG, optimG), (netD, optimD), (netL, optimL)


def save_images(args, sample, recon, step, nrow=8):
    save_path = 'experiments/{}/samples/{}_gen.png'.format(args.name, step)
    save_image(sample, save_path, nrow=nrow, normalize=True)
    if recon is not None:
        save_path = 'experiments/{}/samples/{}_disc.png'.format(args.name, step)
        save_image(recon, save_path, nrow=nrow, normalize=True)
    return


def rand_marginals(args, graph):
    evidence = bn.random_evidence()
    marginals = bn.return_marginals(graph, args.batch_size, evidence)
    marginals = torch.tensor(marginals).cuda()
    return marginals


def pretrain_labeler(args, netL, optimL):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    print ('loading data')
    data_loader = datagen.load_celeba_50k_attrs(args)
    print ('starting pretraining')
    for _ in range(10):
        for i, (data, _, attrs) in enumerate(data_loader):
            data = data.cuda()
            attrs = attrs.cuda().view(args.batch_size, 9)
            attrs = attrs + 1
            attrs[attrs==2] = 1
            output = netL(data)
            loss = bce_loss(output, attrs.float())
            loss.backward()
            optimL.step()
            print (i)
        print ('pretrain loss: ', loss)


def train(args):
    random.seed(8722)
    torch.manual_seed(4565)
    measure_history = deque([0]*3000, 3000)
    convergence_history = []
    prev_measure = 1
    iter = 0
    thresh = 0.5
    
    graph = bn.create_bayes_net()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    lr = args.lr
    iters = args.load_step
    prepare_paths(args)
    u_dist = utils.create_uniform(-1, 1)
    fixed_z = utils.sample_z(u_dist, (args.batch_size, args.z))
    (netG, optimG), (netD, optimD), (netL, optimL) = init_models(args)
    pretrain_labeler(args, netL, optimL)
    data_loader = datagen.load_celeba_50k_attrs(args)
    for epoch in range(args.epochs):
        for i, (data, _, attrs) in enumerate(data_loader):
            data = data.cuda()
            attrs = torch.squeeze(attrs>0).float().cuda()
            """ Labeler """
            netL.zero_grad()
            real_labels = netL(data)
            real_label_loss = bce_loss(real_labels, attrs).mean()
            real_label_loss.backward()
            optimL.step()
            """ Discriminator """
            for p in netD.parameters():
                p.requires_grad = True
            z = utils.sample_z(u_dist, (args.batch_size, args.z))       
            marginals = rand_marginals(args, graph)
            netD.zero_grad()
            with torch.no_grad():
                g_fake = netG(z, marginals)
            _, d_fake = netD(g_fake, marginals)
            _, d_real = netD(data, attrs)
          
            real_loss_d = (d_real - data).abs().mean()
            fake_loss_d = (d_fake - g_fake).abs().mean()

            lossD = real_loss_d - args.k * fake_loss_d
            lossD.backward()
            optimD.step()
            """ Generator """
            for p in netD.parameters():
                p.requires_grad = False
            marginals = rand_marginals(args, graph)
            netG.zero_grad()
            netL.zero_grad()
            z = utils.sample_z(u_dist, (args.batch_size, args.z))       
            g_fake = netG(z, marginals)
            _, d_fake = netD(g_fake, marginals)
            lossG = (d_fake - g_fake).abs().mean()
            
            ## Ok lets penalize distance from marginals (labeler)
            with torch.no_grad():
                marginals = rand_marginals(args, graph)
                z = utils.sample_z(u_dist, (args.batch_size, args.z))       
                g_fake = netG(z, marginals)
            fake_labels = netL(g_fake)
            fake_label_loss = bce_loss(fake_labels, marginals).mean()
            loss_gac = fake_label_loss + lossG
            loss_gac.backward()
            optimG.step(); optimL.step()

            lagrangian = (args.gamma*real_loss_d - fake_loss_d).detach()
            args.k += args.lambda_k * lagrangian
            args.k = max(min(1, args.k), 0)
        
            convg_measure = real_loss_d.item() + lagrangian.abs()
            measure_history.append(convg_measure)
            if iter % args.print_step == 0:
                print ("Iter: {}, D loss: {}, G Loss: {}, AC loss: {}".format(iter,
                       lossD.item(), lossG.item(), fake_label_loss.item()))
                save_images(args, g_fake.detach(), d_real.detach(), iter)
           
            """update training parameters"""
            lr = args.lr * 0.95 ** (iter//3000)
 
            for p in optimG.param_groups + optimD.param_groups:
                p['lr'] = lr

            if iter % 1000 == 0:
                pathG = 'experiments/{}/models/netG_{}.pt'.format(args.name, iter)
                pathD = 'experiments/{}/models/netD_{}.pt'.format(args.name, iter)
                utils.save_model(pathG, netG, optimG, args.k)
                utils.save_model(pathD, netD, optimD, args.k)
            iter += 1


def generative_experiments(args):
    (netG, _,), (netD, _), (netL, _) = init_models(args)
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
