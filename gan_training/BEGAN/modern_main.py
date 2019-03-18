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
    parser.add_argument('--name', default='c-began')
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


def Disc_loss(args, d_real, data, d_fake, g_fake):
    real_loss_d = (d_real - data).abs().mean()
    fake_loss_d = (d_fake - g_fake).abs().mean()
    return (real_loss_d, fake_loss_d)
        

def Gen_loss(args, g_out, g_fake):
    return (g_out - g_fake).abs().mean()

def minus_entropy(x):
    s = torch.sum(x, dim=0)
    #print(s.shape)
    x = x/s
    #print(x.shape)
    #print(x)
    x = x*x.log()

    return x.mean()

def load_part_model(model, path):
    modeldict = model.state_dict()
    pretrained = torch.load(path)['state_dict']
    pretrained = {k:v for k, v in pretrained.items() if k in modeldict}
    modeldict.update(pretrained)
    model.load_state_dict(modeldict)
    return model

def train(args):
    graph = bayes_net.create_bayes_net()

    random.seed(8722)
    torch.manual_seed(4565)
    measure_history = deque([0]*3000, 3000)
    convergence_history = []
    prev_measure = 1
    
    lr = args.lr
    iters = args.load_step
    prepare_paths(args)

    data_loader = datagen.load_celeba_50k_attrs(args)#get_loader(args.data_path, args.batch_size, args.scale, 0)

    z = torch.FloatTensor(args.batch_size, args.z).cuda()
    fixed_z = torch.FloatTensor(args.batch_size, args.z).cuda()
    fixed_z.data.uniform_(-1, 1)    
    fixed_x = None
    iter = 0
    (netG, optimG), (netD, optimD) = init_models(args)

    # load Generator only
    #netG = load_part_model(netG, './experiments/netG_36000.pt')
    #netD = load_part_model(netD, './experiments/netD_36000.pt')
    #
    THRES = 0.5
    for i in range(args.epochs):
        for ii, (data, _, real_atts) in enumerate(data_loader):
            if data.size(0) != args.batch_size:
                continue
            #import pdb; pdb.set_trace()
            real_atts = torch.squeeze(real_atts>0).float().cuda() # convert attr from [-1, 1] -> [0, 1]
            data = data.cuda()
            if fixed_x is None:
                fixed_x = data

            # TRAIN DISCRIMINATOR
            z.data.uniform_(-1, 1).view(args.batch_size, args.z)
            evidence = bayes_net.random_evidence()
            #import pdb; print('Check marginal'); pdb.set_trace()
            marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence)).cuda()
            netD.zero_grad()
            with torch.no_grad():
                g_fake, g_feats = netG(z, marginals)
                #g_feats = (F.sigmoid(g_feats)>0.5).float()
            d_fake = netD(g_fake, (g_feats.detach()>THRES).float())
            d_real = netD(data, real_atts)
           
            real_loss_d, fake_loss_d = Disc_loss(args, d_real, data, d_fake, g_fake)
            lossD = real_loss_d - args.k * fake_loss_d
            lossD.backward()
            optimD.step()

            # TRAIN GENERATOR
            netG.zero_grad() 
            evidence = bayes_net.random_evidence()
            marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence)).cuda()
            g_fake, g_feats = netG(z, marginals)
            g_out= netD(g_fake, (g_feats>THRES).float())
            
            #classification loss
            lossG = Gen_loss(args, g_out, g_fake)
#            import pdb; print('G class_loss: ', g_class_loss); #pdb.set_trace()
            # calculate kl divergence loss between generated batch and marginals
            mu1 = torch.mean(g_feats, dim=0)
            bin1 = Binomial(1, torch.clamp(mu1, 0.001, 0.999))
            bin2 = Binomial(1, torch.clamp(marginals[0], 0.001, 0.999))# because marginals[0]=marginals[1]=...
            kl_loss = 1*kl_divergence(bin1, bin2).mean()
            #neg_entropy = 10*minus_entropy(g_feats)
#            import pdb; print('KL_loss: ', float(kl_loss), mu1); pdb.set_trace()
            (kl_loss+lossG).backward()
            optimG.step()
            if iter % 200 == 0:
                print(marginals[0])
                print(g_feats[:10])
            with torch.no_grad():
                lagrangian = (args.gamma*real_loss_d - fake_loss_d)
                args.k += args.lambda_k * lagrangian
                args.k = max(min(1, args.k), 0)
            
                convg_measure = real_loss_d.item() + lagrangian.abs()
                measure_history.append(convg_measure)
                if iter % args.print_step == 0:
                    print ("Iter: {}, Epoch: {}, \nD loss: {}, G Loss: {}".format(iter, i, 
                        lossD.item(), lossG.item()))
                    print("KL loss {}".format( kl_loss))
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
