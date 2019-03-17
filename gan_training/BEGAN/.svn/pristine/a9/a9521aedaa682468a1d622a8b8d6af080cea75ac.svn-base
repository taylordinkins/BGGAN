import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, args, disc=False):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(Generator, self).__init__()

        self.linear = nn.Linear(self.z, 8*8*self.nc)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
        )
        self.last_conv = nn.Conv2d(self.nc, 3, 3, 1, 1)

    def forward(self, input):
        x = self.linear(input)
        x = F.elu(x)
        x = x.view(self.batch_size, self.nc, 8, 8)
        x = self.conv_block1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_block2(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_block3(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_block4(x)
        x = self.last_conv(x)
        x = torch.tanh(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 1, 1, 0),
        )
        self.pool1 = nn.AvgPool2d(2, 2)

        self.block2 = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.nc, 2*self.nc, 1, 1, 0),
        )
        self.pool2 = nn.AvgPool2d(2, 2)

        self.block3 = nn.Sequential(
            nn.Conv2d(2*self.nc, 2*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(2*self.nc, 2*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(2*self.nc, 3*self.nc, 1, 1, 0),
        )
        self.pool3 = nn.AvgPool2d(2, 2)
        
        self.block4 = nn.Sequential(
            nn.Conv2d(3*self.nc, 3*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(3*self.nc, 3*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
        )
        self.linear1 = nn.Linear(8*8*3*self.nc, 64)
        self.block5 = nn.Sequential(
            nn.Conv2d(3*self.nc, 3*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(3*self.nc, 3*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(3*self.nc, 4*self.nc, 1, 1, 0),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(4*self.nc, 4*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(4*self.nc, 4*self.nc, 3, 1, 1),
            nn.ELU(inplace=True),
        )
        self.linear2 = nn.Linear(8*8*4*self.nc, self.z)

        
    def forward(self, input):
        x = self.block1(input)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        if self.scale == 64:
            x = self.block4(x)
            x = x.view(self.batch_size, 8*8*3*self.nc)
            x = self.linear1(x)
        else:
            x = self.block5(x)
            x = x.view(self.batch_size, 8*8*4*self.nc)
            x = F.elu(self.linear2(x), True)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.enc = Encoder(nc)
        self.dec = Generator(nc, True)
    def forward(self, input):
        return self.dec(self.enc(input))


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
