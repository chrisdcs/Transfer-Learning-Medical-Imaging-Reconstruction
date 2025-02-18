import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model import data_consistency
#import torch.optim as optim


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = ComplexConv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = ComplexConv2D(out_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = ComplexConv2D(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = ComplexReLU()

    def forward(self, x):
        residual = x.clone()
        out = self.relu(self.conv0(x))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)
        out += residual  # Short-skip connection
        return self.relu(out)
    
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ComplexConv2D(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.residual_block = ResidualBlock(out_channels, out_channels)
        self.conv2 = ComplexConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_block(x)
        x = self.conv2(x)
        return x
    
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = ComplexConv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_block = ResidualBlock(out_channels, out_channels)
        self.conv2 = ComplexConv2DTranspose(out_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_block(x)
        x = self.conv2(x)
        
        return x

class PIGANGenerator(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, feature_maps=[64, 128, 256, 512]):
        super(PIGANGenerator, self).__init__()
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i, out_fmaps in enumerate(feature_maps):
            in_fmaps = in_channels if i == 0 else feature_maps[i-1]
            self.encoder_blocks.append(EncoderBlock(in_fmaps, out_fmaps))
            
        self.decoder_blocks = nn.ModuleList()
        for i, out_fmaps in enumerate(reversed(feature_maps[:-1])):
            in_fmaps = feature_maps[-1] if i == 0 else feature_maps[-i-1]
            self.decoder_blocks.append(DecoderBlock(in_fmaps, out_fmaps))
            
        self.decoder_blocks.append(DecoderBlock(out_fmaps, out_fmaps))
        
        self.final_conv = ComplexConv2D(out_fmaps, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, k, mask):
        # Encoder path
        skip_connections = []
        skip_connections.append(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)

        # Decoder path
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i <= 2:
                x = decoder_block(x) + skip_connections[-(i+2)]  # Connect with corresponding encoder block
            else:
                x = decoder_block(x)

        # Final layer
        x = self.final_conv(x) + skip_connections[0]
        # Fx = torch.fft.fft2(x, norm='ortho')
        # Fx = data_consistency(Fx, k, mask)
        # x = torch.fft.ifft2(Fx, norm="ortho")
        return x 


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        
        out_real = F.leaky_relu(x_real, self.negative_slope)
        out_imag = F.leaky_relu(x_imag, self.negative_slope)
        
        return torch.complex(out_real, out_imag)


class PIGANDiscriminator(nn.Module):
    
    def __init__(self, in_channels=1):
        super(PIGANDiscriminator, self).__init__()
        self.conv1 = ComplexConv2D(in_channels, 64, kernel_size=4, stride=2, padding=0)
        self.leaky_relu = ComplexLeakyReLU()

        # Encoder Blocks
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(64, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512)
        )
        
        # Output convolution
        self.conv_final = ComplexConv2D(512, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        # Input convolution
        x = self.conv1(x)
        x = self.leaky_relu(x)

        # Encoder Blocks
        x = self.encoder_blocks(x)

        # Output convolution
        x = self.conv_final(x)
        return torch.real(x) #torch.sigmoid(torch.real(x))  # Output a scalar between 0 and 1
    
    
def MAE_Loss(pred, target):
    return torch.mean(torch.abs(pred - target))

#criterion = nn.BCELoss()

def adversarial_loss(fake_output):
    return -torch.mean(fake_output)

# discriminator_loss

def discriminator_loss(real_output, fake_output):
    #real_loss = criterion(real_output, real_labels)
    #fake_loss = criterion(fake_output, fake_labels)
    #return real_loss + fake_loss
    #d_loss_pos = criterion(real_output, torch.ones_like(real_output))
    #d_loss_neg = criterion(fake_output, torch.zeros_like(fake_output))
    
    return torch.mean(real_output-fake_output)