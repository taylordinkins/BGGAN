import os
import numpy as np
import pandas as pd
import utils
import torch
import torchvision
import pickle
from torchvision import datasets, transforms
from scipy.misc import imread,imresize
from sklearn.model_selection import train_test_split
from glob import glob

ATTR_PATH = '../data/'
KEEP_ATTRS = ['FName', 'Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', 'Wearing_Lipstick', 'Mouth_Slightly_Open', 'Narrow_Eyes']

# get attributes as dataframe for all 200k images
def get_attrs(path):
    attrs = pd.read_csv(path+'list_attr_celeba.csv')
    return attrs[KEEP_ATTRS]

# custom ImageFolder so we can return the attributes
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls

    def __init__(self, path):
        super(ImageFolderWithPaths, self).__init__(path,
                        transform=transforms.Compose([
                        transforms.CenterCrop((108, 108)),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize([.5,.5,.5], [.5,.5,.5])
                        ]))
        # save dataframe for easy access
        self.attrs_df = get_attrs(ATTR_PATH)

    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        fname = os.path.basename(path)
        # attributes lookup
        attrs = self.attrs_df[self.attrs_df['FName'] == fname].drop('FName', axis=1).values
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (attrs,))
        return tuple_with_path

def load_mnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_m/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=100, shuffle=True, **kwargs)
                #batch_size=32, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_notmnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_nm/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_fashion_mnist():
    path = 'data_f'
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_cifar(args):
    path = 'data_c/'
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_celeba_50k(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    path = 'data_c/'
    train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path,
                transform=transforms.Compose([
                    transforms.CenterCrop((108, 108)),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([.5,.5,.5], [.5,.5,.5])
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader

# return dataloader with the attributes
def load_celeba_50k_attrs(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    path = 'data_c/'
    train_img_folder = ImageFolderWithPaths(path)
    train_loader = torch.utils.data.DataLoader(train_img_folder, batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


def load_cifar_hidden(args, c_idx):
    path = './data_c'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    def get_classes(target, labels):
        label_indices = []
        for i in range(len(target)):
            if target[i][1] in labels:
                label_indices.append(i)
        return label_indices

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=False, transform=transform_train)
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, c_idx))
    trainloader = torch.utils.data.DataLoader(train_hidden, batch_size=32,
            shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=False, transform=transform_test)
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, c_idx))
    testloader = torch.utils.data.DataLoader(test_hidden, batch_size=32,
            shuffle=False, **kwargs)
    return trainloader, testloader
