import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import data_consistency
from utils.general import init_seeds


class ComplexConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2D, self).__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        
        out_real = self.conv_re(x_real) - self.conv_im(x_imag)
        out_imag = self.conv_re(x_imag) + self.conv_im(x_real)
        
        return torch.complex(out_real, out_imag)

class ComplexConv2DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2DTranspose, self).__init__()
        self.conv_re = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_im = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        
        out_real = self.conv_re(x_real) - self.conv_im(x_imag)
        out_imag = self.conv_re(x_imag) + self.conv_im(x_real)
        
        return torch.complex(out_real, out_imag)

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        
        out_real = F.relu(x_real)
        out_imag = F.relu(x_imag)
        
        return torch.complex(out_real, out_imag)

class ComplexMaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(ComplexMaxPool2D, self).__init__()
        self.pool_re = nn.MaxPool2d(kernel_size, stride, padding)
        self.pool_im = nn.MaxPool2d(kernel_size, stride, padding)
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        
        out_real = self.pool_re(x_real)
        out_imag = self.pool_im(x_imag)
        
        return torch.complex(out_real, out_imag)
    
class ComplexBatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2D, self).__init__()
        self.bn_re = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_im = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        
        out_real = self.bn_re(x_real)
        out_imag = self.bn_im(x_imag)
        
        return torch.complex(out_real, out_imag)
    
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = self._conv_block(1, 64)
        self.conv2 = self._conv_block(64, 128)
        self.conv3 = self._conv_layer(128, 256)
        self.conv4 = self._conv_layer(256, 128)
        
        self.pool = ComplexMaxPool2D(kernel_size=2, stride=2)
        
        self.up_conv1 = self._upconv_block(128, 128)
        self.conv5 = self._conv_layer(256, 128)
        self.conv6 = self._conv_layer(128, 64)
        
        self.up_conv2 = self._upconv_block(64, 64)
        
        self.conv7 = self._conv_block(128, 64)
        
        self.conv8 = self._conv_layer(64, 1, kernel_size=1, padding=0)
        
        
    def _upconv_block(self, in_channels, out_channels):
        return ComplexConv2DTranspose(in_channels, out_channels, kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            ComplexConv2D(in_channels, out_channels, kernel_size=3, padding=1),
            ComplexReLU(),
            ComplexConv2D(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexReLU(),
            ComplexBatchNorm2D(out_channels)
        )
    
    def _conv_layer(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            ComplexConv2D(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            ComplexReLU(),
            ComplexBatchNorm2D(out_channels)
        )
    
    def forward(self, x, k, mask):
        x1 = self.conv1(x)
        x1_cat = x1.clone()
        x1 = self.pool(x1)
        x2 = self.conv2(x1)
        x2_cat = x2.clone()
        x2 = self.pool(x2)
        x3 = self.conv3(x2)
        
        x4 = self.conv4(x3)
        
        x4 = self.up_conv1(x4)
        x4 = torch.cat([x4, x2_cat], dim=1)
        
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        x6 = self.up_conv2(x6)
        x6 = torch.cat([x6, x1_cat], dim=1)
        
        x7 = self.conv7(x6)
        
        x8 = self.conv8(x7)
        
        Fx = torch.fft.fft2(x8, norm="ortho")
        Fx = data_consistency(Fx, k, mask)
        x = torch.fft.ifft2(Fx, norm="ortho")
        
        return x
    
