import torch
import torch.nn as nn
import os
from utils.model import data_consistency
#from utils.UMRI_model import *

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.dataset import *

from utils.general import init_seeds

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

    def forward(self, x, k, m, MD=False):
        if MD:
            feats = []
        for i in range(self.nc):
            
            #real = self.r_conv_blocks[i](x.real) - self.i_conv_blocks[i](x.imag)
            #imag = self.r_conv_blocks[i](x.imag) + self.i_conv_blocks[i](x.real)
            #x_cnn = torch.complex(real, imag)
            x_cnn = x.clone()
            for idx, layer in enumerate(self.conv_blocks[i]):
                x_cnn = layer(x_cnn)
                if MD and idx == 5:
                    feat = x_cnn.clone()
                    feats.append(feat)
            # x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            
            new_k = torch.fft.fft2(x, norm="ortho")
            new_k = data_consistency(new_k, k, m)
            x = torch.fft.ifft2(new_k, norm="ortho")
        if MD:
            return x, feats
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

    def forward(self, x, k, m, anatomy, MD=False):
        if MD:
            feats = []
        for i in range(self.nc):
            
            #real = self.r_conv_blocks[i](x.real) - self.i_conv_blocks[i](x.imag)
            #imag = self.r_conv_blocks[i](x.imag) + self.i_conv_blocks[i](x.real)
            #x_cnn = torch.complex(real, imag)
            x_cnn = x.clone()
            for j in range(2 * self.nd-1):
                x_cnn = self.conv_blocks[i][j](x_cnn)
                if MD and j == 5:
                    feat = x_cnn.clone()
                    feats.append(feat)
                if j % 2 == 0 and j < 8:
                    x_cnn = self.ASPINs[anatomy][i][j//2](x_cnn)
            
            #x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)
            
            new_k = torch.fft.fft2(x, norm="ortho")
            new_k = data_consistency(new_k, k, m)
            x = torch.fft.ifft2(new_k, norm="ortho")
        if MD:
            return x, feats
        return x


def attention_loss(feat1, feat2):
    O1 = torch.sum(torch.abs(feat1), dim=1, keepdim=True)
    O2 = torch.sum(torch.abs(feat2), dim=1, keepdim=True)
    
    # normalize O1 and O2 other than the batch dimension
    O1_norm = O1 / torch.norm(O1, dim = (2, 3), keepdim=True)
    O2_norm = O2 / torch.norm(O2, dim = (2, 3), keepdim=True)
    
    return torch.sum(torch.abs(O1_norm - O2_norm))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
acc = 5
mask = 'radial'
dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 
                          'data/knee/knee_singlecoil_train.mat'], 
                         acc=acc, mask=mask)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

brain_model = DnCn()
knee_model = DnCn()

brain_model.to(device)
knee_model.to(device)

brain_model.load_state_dict(torch.load(os.path.join("universal_MRI", "brain", 
                                                    f"{mask}_{acc}", 'model_100.pth')))
knee_model.load_state_dict(torch.load(os.path.join("universal_MRI", "knee", 
                                                   f"{mask}_{acc}", 'model_100.pth')))

universal_model = UDnCn(['brain', 'knee'])

universal_model.load_state_dict(torch.load(os.path.join("universal_MRI", "universal", f"{mask}_{acc}", 'model_200.pth')))
universal_model.to(device)

optim = torch.optim.Adam(universal_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.95)

n_epochs = 200

folder = os.path.join("universal_MRI", "universal", f"{mask}_{acc}_MD")
if not os.path.exists(folder):
    os.makedirs(folder)

for epoch in range(1, n_epochs+1):
    
    PSNR_list = []
    loss_list = []
    
    for i, data in enumerate(loader):
        
        im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
        
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)    
        
        optim.zero_grad()
        output, feats = universal_model(im_und, k_und, mask, anatomy[0], MD=True)
        output = torch.abs(output).clamp(0, 1)
        
        if anatomy[0] == 'brain':
            _, feats_ind = brain_model(im_und, k_und, mask, MD=True)
        elif anatomy[0] == 'knee':
            _, feats_ind = knee_model(im_und, k_und, mask, MD=True)
        
        img_gnd = torch.abs(img_gnd)
        
        loss = torch.sum(torch.square(output - img_gnd))
        AL = 0
        for feat1, feat2 in zip(feats, feats_ind):
            AL += attention_loss(feat1, feat2)
        #print(loss.data, AL.data)
        #break
        loss += AL * 1e-3
        
        loss.backward()
        optim.step()
        
        loss_list.append(loss.item())
        
        for j in range(output.shape[0]):
            PSNR_list.append(psnr(output[j].cpu().detach().numpy(), img_gnd[j].cpu().detach().numpy()))
        if (i+1) % 100 == 0:
            print(i+1, loss.item())
    scheduler.step()
    avg_l = np.mean(loss_list)
    avg_p = np.mean(PSNR_list)
    epoch_data = ' [Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
            (epoch, n_epochs, avg_l, avg_p)
    print(epoch_data)
    
    if (epoch) % 10 == 0:
        torch.save(universal_model.state_dict(), os.path.join(folder, 'model_%d.pth' % (epoch)))
        print('Model saved\n')