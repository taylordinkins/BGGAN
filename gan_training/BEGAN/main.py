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
from models import Generator, Discriminator, AttributeDetector
from dataloader import *

import bayes_net


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
    parser.add_argument('--data_path', default='/nfs/guille/wong/wonglab2/datasets/data_c/val/val')
    parser.add_argument('--load_step', default=0, type=int)
    parser.add_argument('--print_step', default=100, type=int)
    parser.add_argument('--resume_G', default=None)
    parser.add_argument('--resume_D', default=None)
    parser.add_argument('--evidence_exp', default=None)
    parser.add_argument('--data_parallel', default=None)
    parser.add_argument('--samples', default=None)
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

def load_feature_detector(args):
    print('Loading feature detector...\n')
    fdet = AttributeDetector(args).cuda()
    fdet.load_state_dict(torch.load('../saved_models/celeba/netFDET_best_r2'))
    for p in fdet.parameters():
        p.requires_grad = False
    fdet.eval()

    return fdet

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

def evidence_test(args):
    torch.manual_seed(8734)
    print('Evidence test...\n')
    
    print('Loading models...\n')
    (netG, optimG), (netD, optimD) = init_models(args)
    fdet = load_feature_detector(args)
    graph = bayes_net.create_bayes_net()
    evidence = bayes_net.evidence_query(['Young', 'Glasses'], [1, 1])
    marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence)).cuda()
    z = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
    bce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    
    for p in netD.parameters():
        p.requires_grad = False
    for pp in netG.parameters():
        p.requires_grad = False


    g_fake = netG(z, marginals)
    g_attrs = torch.sigmoid(fdet(g_fake))
    classif_loss = torch.mean(torch.min(g_attrs, 1-g_attrs))
    g_attrs = torch.round(g_attrs).mean(0)
    dist_loss = mse_loss(g_attrs, marginals[0])
    classif_coef = math.exp(-(3000/(iter+1)))
    dist_coef = math.exp(-(3000/(iter+1)))

    print('Distribution Loss', dist_loss.cpu().item())
    print('Classification Loss', classif_loss.cpu().item())
    print()

    print('Saving images...\n')
    save_images(args, g_fake, None, 'test')


        


def train(args):
    random.seed(8722)
    torch.manual_seed(42)
    measure_history = deque([0]*3000, 3000)
    convergence_history = []
    prev_measure = 1
    
    lr = args.lr
    iters = args.load_step
    prepare_paths(args)

    data_loader = get_loader(args.data_path, args.batch_size, args.scale, 0)

    # z = torch.FloatTensor(args.batch_size, args.z).cuda()
    fixed_z = torch.FloatTensor(args.batch_size, args.z).cuda()
    fixed_z.data.uniform_(-1, 1)    
    fixed_x = None
    iter = 0
    (netG, optimG), (netD, optimD) = init_models(args)

    print('Initializing weights...\n')
    netG.apply(weights_init)
    netD.apply(weights_init)
    graph = bayes_net.create_bayes_net()
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    fdet = load_feature_detector(args)
    if torch.cuda.device_count() > 1 and args.data_parallel is not None:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        fdet = nn.DataParallel(fdet)
    for i in range(args.epochs):
        for _, data in enumerate(data_loader):
            if data.size(0) != args.batch_size:
                continue
            #print (iter)
            data = data.cuda()
            if fixed_x is None:
                fixed_x = data
            z = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            evidence = bayes_net.random_evidence()
            marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence), requires_grad=True).cuda()

            netD.zero_grad()
            netG.zero_grad()
            with torch.no_grad():
                g_fake = netG(z, marginals)
            d_fake = netD(g_fake)
            d_real = netD(data)
           
            real_loss_d, fake_loss_d = Disc_loss(args, d_real, data, d_fake, g_fake)
            lossD = real_loss_d - args.k * fake_loss_d
            lossD.backward()
            optimD.step()
            
            z = torch.randn(args.batch_size, args.z, requires_grad=True).cuda()
            evidence = bayes_net.random_evidence()
            marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence), requires_grad=True).cuda()

            g_fake = netG(z, marginals)
            g_out = netD(g_fake)
            g_attrs = torch.sigmoid(fdet(g_fake))
            classif_loss = torch.mean(torch.min(g_attrs, 1-g_attrs))
            g_attrs = torch.round(g_attrs).mean(0)
            dist_loss = mse_loss(g_attrs, marginals[0])
            classif_coef = math.exp(-(10000/(iter+1)))
            dist_coef = math.exp(-(8000/(iter+1)))
            lossG = Gen_loss(args, g_out, g_fake) + classif_coef*classif_loss + dist_coef*dist_loss
            lossG.backward()
            optimG.step()
            
            lagrangian = (args.gamma*real_loss_d - fake_loss_d).detach()
            args.k += args.lambda_k * lagrangian
            args.k = max(min(1, args.k), 0)
        
            convg_measure = real_loss_d.item() + lagrangian.abs()
            measure_history.append(convg_measure)

            
           
            """update training parameters"""
            lr = args.lr * 0.95 ** (iter//3000)
 
            for p in optimG.param_groups + optimD.param_groups:
                p['lr'] = lr

            if iter % 10 == 0:
                print ("Iter: {}, Epoch: {}, D loss: {}, G Loss: {}, \nClass Loss: {}, Dist Loss: {}".format(iter, i, 
                    lossD.item(), lossG.item(), classif_loss.item(), dist_loss.item()))
                print()

            if iter % 250 == 0:
                print('Saving images...\n')
                save_images(args, g_fake.detach(), d_real.detach(), iter)

            if iter % 1000 == 0 and iter > 0:
                print('Saving models...\n')
                pathG = 'experiments/{}/models/netG_began.pt'.format(args.name)
                pathD = 'experiments/{}/models/netD_began.pt'.format(args.name)
                utils.save_model(pathG, netG, optimG)
                utils.save_model(pathD, netD, optimD)
        
            iter += 1

def train_samples(args):
    random.seed(8722)
    torch.manual_seed(42)
    measure_history = deque([0]*3000, 3000)
    convergence_history = []
    prev_measure = 1
    
    fdet = load_feature_detector(args)
    graph = bayes_net.create_bayes_net()
    evidence = bayes_net.evidence_query(['Young', 'Glasses'], [1, 1])

    bce_loss = torch.nn.BCEWithLogitsLoss()

    lr = args.lr
    iters = args.load_step
    prepare_paths(args)
    u_dist = utils.create_uniform(-1, 1)
    data_loader = datagen.load_celeba_50k(args)
    fixed_z = utils.sample_z(u_dist, (args.batch_size, args.z))
    iter = 0
    (netG, optimG), (netD, optimD) = init_models(args)
    print('Initializing weights...\n')
    netG.apply(weights_init)
    netD.apply(weights_init)
    for i in range(args.epochs):
        for i, (data, _) in enumerate(data_loader):
            data = data.cuda()
            z = utils.sample_z(u_dist, (args.batch_size, args.z))       
            z = z.view(args.batch_size, args.z)
            evidence = bayes_net.random_evidence()
            marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence))
            mdist = torch.distributions.Bernoulli(marginals)
            attr_samples = mdist.sample().cuda().requires_grad_(True)
            netD.zero_grad()
            with torch.no_grad():
                g_fake = netG(z, attr_samples)
            d_fake = netD(g_fake)
            d_real = netD(data)
           
            real_loss_d, fake_loss_d = Disc_loss(args, d_real, data, d_fake, g_fake)
            lossD = real_loss_d - args.k * fake_loss_d
            lossD.backward()
            optimD.step()
            

            z = utils.sample_z(u_dist, (args.batch_size, args.z))       
            z = z.view(args.batch_size, args.z)
            evidence = bayes_net.random_evidence()
            marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence))
            mdist = torch.distributions.Bernoulli(marginals)
            attr_samples = mdist.sample().cuda().requires_grad_(True)
            
            netG.zero_grad()
            g_fake = netG(z, attr_samples)
            g_out = netD(g_fake)
            fake_preds = torch.sigmoid(fdet(fake))
            g_attrs = bce_loss(fake_preds, attr_samples)
            classif_coef = math.exp(-(4500/(iter+1)))
            lossG = Gen_loss(args, g_out, g_fake) + classif_coef*g_attrs
            lossG.backward()
            optimG.step()

            lagrangian = (args.gamma*real_loss_d - fake_loss_d).detach()
            args.k += args.lambda_k * lagrangian
            args.k = max(min(1, args.k), 0)
        
            convg_measure = real_loss_d.item() + lagrangian.abs()
            measure_history.append(convg_measure)
           
            """update training parameters"""
            lr = args.lr * 0.95 ** (iter//3000)
 
            for p in optimG.param_groups + optimD.param_groups:
                p['lr'] = lr

            if iter % 10 == 0:
                print ("Iter: {}, Epoch: {}, D loss: {}, G Loss: {}".format(iter, i, 
                    lossD.item(), lossG.item()))
            
            if iter % 50 == 0:
                print('Saving images...\n')
                save_images(args, g_fake.detach(), d_real.detach(), iter)

            if iter % 1000 == 0:
                pathG = 'experiments/{}/models/netG_samples.pt'.format(args.name)
                pathD = 'experiments/{}/models/netD_samples.pt'.format(args.name)
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
    if args.evidence_exp:
        args.resume_D = 1
        args.resume_G = 1
        evidence_test(args)
    else:
        if samples:
            train_samples(args)
        else:
            train(args)
    #generative_experiments(args)
