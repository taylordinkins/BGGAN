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
from models import Generator, Discriminator, AttributeDetector, DiscriminatorWithAttributes, BasicDiscriminator, NGenerator
from dataloader import *

import bayes_net
import datagen

from enum import Enum

KEEP_ATTRS = ['FName', 'Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']

class AttrEnum(Enum):
    Young = 0
    Male = 1
    Eyeglasses = 2
    Bald = 3
    Mustache = 4
    Smiling = 5
    Wearing_Lipstick = 6
    Mouth_Slightly_Open = 7
    Narrow_Eyes = 8

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
    parser.add_argument('--sampling', default=None)
    args = parser.parse_args()
    return args


def prepare_paths(args):
    utils.create_if_empty('./experiments')
    utils.create_if_empty('./experiments/{}/'.format(args.name))
    utils.create_if_empty('./experiments/{}/models'.format(args.name))
    utils.create_if_empty('./experiments/{}/samples'.format(args.name))
    utils.create_if_empty('./experiments/{}/params'.format(args.name))
    utils.create_if_empty('./experiments/{}/'.format(args.name+'-reals'))
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


def save_images(args, sample, recon, step, nrow=0):
    save_path = 'experiments/{}/{}_gen.png'.format(args.name, step)
    print('Saving into ', save_path)
    print(sample.shape)
    save_image(sample, save_path, nrow=nrow, normalize=True)
    if recon is not None:
        save_path = 'experiments/{}/{}_disc.png'.format(args.name, step)
        save_image(recon, save_path, nrow=nrow, normalize=True)
    return

def save_real_images(args, sample, step, nrow=0):
    save_path = 'experiments/{}/{}_disc.png'.format(args.name+'-reals', step)
    save_image(sample, save_path, nrow=nrow, normalize=True)
    return

def load_part_model(model, path):
    modeldict = model.state_dict()
    pretrained = torch.load(path)['state_dict']
    pretrained = {k:v for k, v in pretrained.items() if k in modeldict}
    modeldict.update(pretrained)
    model.load_state_dict(modeldict)
    return model

def load_feature_detector(args):
    print('Loading feature detector...\n')
    fdet = AttributeDetector(args).cuda()
    fdet.load_state_dict(torch.load('/nfs/stak/users/dinkinst/Homework/cs535/BGGAN/gan_training/saved_models/celeba/netFDET_best_r2'))
    for p in fdet.parameters():
        p.requires_grad = False
    fdet.eval()

    return fdet

def inf_gen_attrs(data_gen):
    while True:
        for images, targets, img_attrs in data_gen:
            images.requires_grad_(True)
            images = images.cuda()
            #img_attrs[img_attrs==-1] = 0
            yield (images, targets, img_attrs)

def match_attrs(img_attrs, evs_attrs, evs_vals):
    attrs = img_attrs.numpy()[0]
    for i, evname in enumerate(evs_attrs):
        # print(AttrEnum[evname].value)
        # print(evs_vals)
        if attrs[AttrEnum[evname].value] != evs_vals[i]:
            return False

    return True

def get_reals(args, evs_attrs, evs_vals):
    data_loader = datagen.load_celeba_50k_train(args)
    count = 0
    train_loader = inf_gen_attrs(data_loader)

    while count < 2000:
        imgs, _, img_attrs = next(train_loader)
        for i, img in enumerate(imgs):
            if match_attrs(img_attrs[i], evs_attrs, evs_vals):
                save_real_images(args, img, count)
                count += 1
                print('Count:', count)

    return




def query(args):
    prepare_paths(args)
    evs_attrs = ['Male', 'Smiling']
    evs_vals = [1, 1]
    # evs_attrs = ['Wearing_Lipstick', 'Young']
    # evs_vals = [1,1]

    #get_reals(args, evs_attrs, evs_vals)

    evidence = bayes_net.evidence_query(evs_attrs, evs_vals)
    
    #(netG, optimG), (netD, optimD) = init_models(args)

    # netG = NGenerator(args)
    # netG.load_state_dict(torch.load('./model_backups/netG_90000.pt')['state_dict'])
    netG = Generator(args)
    netG.load_state_dict(torch.load('./model_backups/netG_samples.pt')['state_dict'])
    netG = netG.cuda()
    graph = bayes_net.create_bayes_net()
    #fdet = load_feature_detector(args)
    
    total_gens = 0
    for j in range(2000//args.batch_size + 1):
        z = torch.randn(args.batch_size, args.z).cuda()
        marginals = torch.tensor(bayes_net.return_marginals(graph, args.batch_size, evidence)).cuda()
        if args.sampling:
            mdist = torch.distributions.Bernoulli(marginals)
            attr_samples = mdist.sample().cuda()
            with torch.no_grad():
                g_fake = netG(z, attr_samples)
                for i, gimg in enumerate(g_fake):
                    save_images(args, gimg, None, total_gens)
                    total_gens += 1
                #save_images(args, g_fake, None, 0, 8)
                #pred_feats = fdet(g_fake)
        else:
            with torch.no_grad():
                g_fake = netG(z, marginals)
                
                #pred_feats = fdet(g_fake)
        # print('Marginals', marginals[0].cpu().detach().numpy())
        # #preds_rnd = torch.round(torch.sigmoid(pred_feats)).mean(0).cpu().detach().numpy()
        # print('Preds', preds_rnd)
        # print('Subtract', marginals[0].cpu().detach().numpy() - preds_rnd)
        
        
    
if __name__ == "__main__":
    args = load_args()
query(args)