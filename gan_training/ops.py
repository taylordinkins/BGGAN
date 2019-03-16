import torch
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd

import utils

def batch_zero_grad(modules):
    for module in modules:
        module.zero_grad()


def batch_update_optim(optimizers):
    for optimizer in optimizers:
        optimizer.step()


def free_params(modules):
    if type(modules) is not list:
        for p in modules.parameters():
            p.requires_grad = False
    else:        
        for module in modules:
            for p in module.parameters():
                p.requires_grad = False


def frozen_params(modules):
    if type(modules) is not list:
        for p in modules.parameters():
            p.requires_grad = False
    else:
        for module in modules:
            for p in module.parameters():
                p.requires_grad = False


def pretrain_loss(encoded, noise):
    mean_z = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_z, mean_e)

    cov_z = torch.matmul((noise-mean_z).transpose(0, 1), noise-mean_z)
    cov_z /= 999
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 999
    cov_loss = F.mse_loss(cov_z, cov_e)
    return mean_loss, cov_loss


def grad_penalty_1dim(args, netD, data, fake):
    alpha = torch.randn(args.batch_size, 1, requires_grad=True).cuda()
    alpha = alpha.expand(data.size()).cuda()
    interpolates = alpha * data + ((1 - alpha) * fake).cuda()
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.l
    return gradient_penalty


def grad_penalty_3dim(args, netD, data, fake, resnet=True):
    alpha = torch.randn(args.batch_size, 1, requires_grad=True).cuda()
    alpha = alpha.expand(args.batch_size, data.nelement()//args.batch_size)
    alpha = alpha.contiguous().view(args.batch_size, 3, 64, 64)
    interpolates = alpha * data + ((1 - alpha) * fake).cuda()
    if resnet:
        disc_interpolates, _ = netD(interpolates)
    # only unpack 1 value
    else:
        disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.l
    return gradient_penalty


def consistency_term(args, netD, data, Mtag=0):
    d1, d_1 = netD(data, xn=True)
    d2, d_2 = netD(data, xn=True)

    consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * (d_1 - d_2).norm(2, dim=1) - Mtag
    return consistency_term.mean()
