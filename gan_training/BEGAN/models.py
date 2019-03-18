import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class Decoder(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(Decoder, self).__init__()

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


class Generator(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(Generator, self).__init__()

        self.linear = nn.Linear(self.z, 8*8*self.nc - 1000)
        self.linear2 = nn.Linear(9, 1000)
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

    def forward(self, input, marginals):
        x = self.linear(input)
        x = F.elu(x)
        y = self.linear2(marginals.contiguous())
        y = F.elu(y)
        x = torch.cat((x, y), 1)
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
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.enc = Encoder(args)
        self.dec = Generator(args)
    def forward(self, input, attrs):
        return self.dec(self.enc(input), attrs)

class BasicDiscriminator(nn.Module):
    def __init__(self, args):
        super(BasicDiscriminator, self).__init__()
        self.enc = Encoder(args)
        self.dec = Decoder(args)
    def forward(self, input):
        return self.dec(self.enc(input))


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)






DIM=64
OUTPUT_DIM=64*64*3

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True,  stride=1, bias=True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = (kernel_size - 1)//2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = input_depth // self.block_size_sq
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width,
                self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(
                batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, args, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size=kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size=kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output
        
class AttributeDetector(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(AttributeDetector, self).__init__()
        self.dim = self.nc
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


class EncoderWithAttributes(nn.Module):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        super(EncoderWithAttributes, self).__init__()
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
        self.linear3 = nn.Linear(8*8*3*self.nc, 9)
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
        y = x
        if self.scale == 64:
            x = self.block4(x)
            x = x.view(self.batch_size, 8*8*3*self.nc)
            y = self.linear3(x)
            x = self.linear1(x)
        else:
            x = self.block5(x)
            x = x.view(self.batch_size, 8*8*4*self.nc)
            x = F.elu(self.linear2(x), True)
        return x, y

class DiscriminatorWithAttributes(nn.Module):
    def __init__(self, args):
        super(DiscriminatorWithAttributes, self).__init__()
        self.enc = EncoderWithAttributes(args)
        self.dec = Decoder(args)
    def forward(self, input):
        enc_z, enc_y = self.enc(input)
        return self.dec(enc_z), enc_y
