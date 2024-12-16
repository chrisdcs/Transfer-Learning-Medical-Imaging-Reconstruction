import torch
import torch.nn as nn
import os
from utils.model import data_consistency


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.dataset import *

from utils.general import init_seeds


#def lrelu():
#    return nn.LeakyReLU(0.01, inplace=True)


#def relu():
#    return nn.ReLU(inplace=True)


class lrelu(nn.Module):
    def __init__(self):
        super(lrelu, self).__init__()
        self.relu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        return self.relu(x.real) + 1j*self.relu(x.imag)
    
class relu(nn.Module):
    def __init__(self):
        super(relu, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x.real) + 1j*self.relu(x.imag)

class complex_conv(nn.Module):
    
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1, dilation=1, bias=True):
        
        super(complex_conv, self).__init__()
        self.rconv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=padding, stride=stride, dilation=dilation, bias=bias)
        self.iconv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=padding, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        out_real = self.rconv(x_real) - self.iconv(x_imag)
        out_imag = self.rconv(x_imag) + self.iconv(x_real)
        return out_real + 1j*out_imag
    
def conv_block(n_ch, nd, nf=16, ks=3, dilation=1, bn=False, nl='lrelu', n_out=None):

    # convolution dimension (2D or 3D)
    conv = complex_conv#nn.Conv2d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)

class DnCn(nn.Module):
    def __init__(self, n_channels=1, nc=5, nd=5, **kwargs):
        super(DnCn, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        #dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            #dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        #self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            
            #real = self.r_conv_blocks[i](x.real) - self.i_conv_blocks[i](x.imag)
            #imag = self.r_conv_blocks[i](x.imag) + self.i_conv_blocks[i](x.real)
            #x_cnn = torch.complex(real, imag)
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            
            new_k = torch.fft.fft2(x, norm="ortho")
            new_k = data_consistency(new_k, k, m)
            x = torch.fft.ifft2(new_k, norm="ortho")

        return x
    
    
def conv_block_list(n_ch, nd, nf=16, ks=3, dilation=1, bn=False, nl='lrelu', n_out=None):

    # convolution dimension (2D or 3D)
    conv = complex_conv#nn.Conv2d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.ModuleList(layers)

class complex_ASPIN(nn.Module):
    def __init__(self, nf=16):
        super(complex_ASPIN, self).__init__()
        self.r_IN = nn.InstanceNorm2d(nf, affine=True)
        self.i_IN = nn.InstanceNorm2d(nf, affine=True)
    def forward(self, x):
        return self.r_IN(x.real) + 1j*self.i_IN(x.imag)

def create_ASPIN(nf=16, nc=5, nd=5):
    ASPIN_blocks = []
    
    for j in range(nc):
        blocks = []
        for i in range(nd-1):
            blocks.append(complex_ASPIN(nf))
        blocks = nn.ModuleList(blocks)
        ASPIN_blocks.append(blocks)
    ASPIN_blocks = nn.ModuleList(ASPIN_blocks)
    
    return ASPIN_blocks


class UDnCn(nn.Module):
    def __init__(self, anatomies, n_channels=1, nc=5, nd=5, **kwargs):
        super(UDnCn, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        #dcs = []

        conv_layer = conv_block_list
        
        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            #dcs.append(DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        
        self.ASPINs = nn.ModuleDict(
            {anatomy: create_ASPIN() for anatomy in anatomies}
        )

    def add_anatomy(self, anatomy):
        assert anatomy not in self.ASPINs
        self.ASPINs[anatomy] = create_ASPIN()

    def forward(self, x, k, m, anatomy):
        for i in range(self.nc):
            
            #real = self.r_conv_blocks[i](x.real) - self.i_conv_blocks[i](x.imag)
            #imag = self.r_conv_blocks[i](x.imag) + self.i_conv_blocks[i](x.real)
            #x_cnn = torch.complex(real, imag)
            x_cnn = x.clone()
            for j in range(2 * self.nd-1):
                x_cnn = self.conv_blocks[i][j](x_cnn)
                if j % 2 == 0 and j < 8:
                    x_cnn = self.ASPINs[anatomy][i][j//2](x_cnn)
            
            #x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            
            new_k = torch.fft.fft2(x, norm="ortho")
            new_k = data_consistency(new_k, k, m)
            x = torch.fft.ifft2(new_k, norm="ortho")

        return x