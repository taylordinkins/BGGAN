import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, args, gen_feature=True):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(Generator, self).__init__()

        self.linear = nn.Linear(self.z, 8*8*self.nc-1000)
        self.linear_m = nn.Linear(9, 1000)
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
        self.gen_feats = gen_feature
        if gen_feature:
           self.feat = nn.Sequential(nn.Linear(8*8*self.nc, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 9),
                                    nn.Sigmoid())
            


    def forward(self, input, marginals):
        x = self.linear(input)
        x = F.elu(x)
        x_m = F.elu(self.linear_m(marginals.contiguous()))
        x = torch.cat((x, x_m), 1)
        x = x.view(self.batch_size, self.nc, 8, 8)
        x = self.conv_block1(x)
        if self.gen_feats:
            feats = self.feat(x.contiguous().view(self.batch_size, -1))
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_block2(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_block3(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_block4(x)
        x = self.last_conv(x)
        x = torch.tanh(x)

        if self.gen_feats:
            return x, feats
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
        self.feat_score1 = nn.Linear(8*8*3*self.nc, 9)

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
        self.feat_score2 = nn.Linear(8*8*4*self.nc, 9)

        
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
            feat = self.feat_score1(x)
            x = self.linear1(x)
        else:
            x = self.block5(x)
            x = x.view(self.batch_size, 8*8*4*self.nc)
            feat = self.feat_score2(x)
            x = F.elu(self.linear2(x), True)
        return x, feat
    
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.enc = Encoder(args)
        self.dec = Generator(args, gen_feature=False)

    def forward(self, input):#,attrs=None):
        h, feat = self.enc(input)
        #print(feat)
        x = self.dec(h, feat)
        #x = self.dec(h, attrs)
        return x, feat

"""
class AttributeDetector(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(AttributeDetector, self).__init__()
        dim = self.dim
        self.conv1 = MyConvo2d(3, dim, 3, he_init = False)
        self.rb1 = ResidualBlock(args, dim, 2*dim, 3, resample='down', hw=dim)
        self.rb2 = ResidualBlock(args, 2*dim, 4*dim, 3, resample='down', hw=dim//2)
        self.rb3 = ResidualBlock(args, 4*dim, 8*dim, 3, resample='down', hw=dim//4)
        self.rb4 = ResidualBlock(args, 8*dim, 8*dim, 3, resample='down', hw=dim//8)
        self.ln1 = nn.Linear(4*4*8*dim, 9)

    def forward(self, input):
        output = input.contiguous()
        output = output.view(-1, 3, 64, 64)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4*4*8*self.dim)
        output = self.ln1(output)
        output = output.view(output.size(0), -1)
        #print(output.shape)
        return output
"""

def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
