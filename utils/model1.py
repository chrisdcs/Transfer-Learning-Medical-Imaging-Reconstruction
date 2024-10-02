
from mamba_ssm import Mamba
import numpy as np

from utils.model import *

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
    def forward(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        #x_norm = x_flat
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        
        return out

class Complex_Mamba_Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Complex_Mamba_Block, self).__init__()
        
        n_feats = kwargs['n_feats']
        n_convs = kwargs['n_convs']
        k_size = kwargs['k_size']
        padding = kwargs['padding']
        self.padding = padding
        
        self.soft_thr = nn.Parameter(torch.Tensor([0.002]),requires_grad=True)
        Rconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                 [MambaLayer(n_feats, n_feats, 4, 2)]
                  #nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                  
                 #[MambaLayer(n_feats, n_feats, 4, 2) for i in range(n_convs-2)]
                 #[nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
                 
        
        self.Rconvs = nn.ModuleList(Rconvs)
        
        Iconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                 [MambaLayer(n_feats, n_feats, 4, 2)]
                  #nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)]+ \
                 
                 #[nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
        self.Iconvs = nn.ModuleList(Iconvs)
        
        
        RconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)]+\
                  [MambaLayer(n_feats, n_feats, 4, 2)]
                   #nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                  #[MambaLayer(n_feats, n_feats, 4, 2)]
                  #[nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
                  
        self.RconvsT = nn.ModuleList(RconvsT)
        
        IconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)]+\
                  [MambaLayer(n_feats, n_feats, 4, 2)]
                   #nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                  
                  #[nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
                  
        self.IconvsT = nn.ModuleList(IconvsT)
        
        self.act = sigma_activation(0.001)
        self.act_der = sigma_derivative(0.001)
    
    
    def gradient(self, forward_cache, gamma):
        soft_thr = torch.abs(self.soft_thr) * gamma
        g = forward_cache[-1]
        
        # compute gradient of smoothed regularization
        norm_g = torch.norm(g, dim = 1, keepdim=True)
        denominator = torch.where(norm_g > soft_thr, norm_g, soft_thr)
        out = torch.div(g, denominator)
        
        for i in range(len(forward_cache)-1, 0, -1):
            out_real, out_imag = out.real, out.imag
            #tmp_real, tmp_imag = self.act_der(forward_cache[i-1].clone().real), self.act_der(forward_cache[i-1].clone().imag)
            #out_real_next = F.conv_transpose2d(out_real, self.Rconvs[i].weight, padding=self.padding) - F.conv_transpose2d(out_imag, self.Iconvs[i].weight, padding=self.padding)
            #out_img_next = F.conv_transpose2d(out_real, self.Iconvs[i].weight, padding=self.padding) + F.conv_transpose2d(out_imag, self.Rconvs[i].weight, padding=self.padding)
            out_real_next = self.RconvsT[i](out_real) - self.IconvsT[i](out_imag)
            out_img_next = self.IconvsT[i](out_real) + self.RconvsT[i](out_imag)
            out = out_real_next + 1j * out_img_next
            #out = out_real_next * tmp_real - out_img_next * tmp_imag + 1j * (out_real_next * tmp_imag + out_img_next * tmp_real)
        out_real, out_imag = out.real, out.imag
        #out_real_next = F.conv_transpose2d(out_real, self.Rconvs[0].weight, padding=self.padding) - F.conv_transpose2d(out_imag, self.Iconvs[0].weight, padding=self.padding)
        #out_img_next = F.conv_transpose2d(out_real, self.Iconvs[0].weight, padding=self.padding) + F.conv_transpose2d(out_imag, self.Rconvs[0].weight, padding=self.padding)
        out_real_next = self.RconvsT[0](out_real) - self.IconvsT[0](out_imag)
        out_img_next = self.IconvsT[0](out_real) + self.RconvsT[0](out_imag)
        out = out_real_next + 1j * out_img_next
        
        return out


    
    def forward(self, x):
        cache = []
        
        for i, (Rconv, Iconv) in enumerate(zip(self.Rconvs, self.Iconvs)):
            x_real, x_imag = x.real, x.imag
            if i == 0:
                x_real_next = Rconv(x_real) - Iconv(x_imag)
                x_imag_next = Rconv(x_imag) + Iconv(x_real)
            else:
                # x_real, x_imag = self.act(x_real), self.act(x_imag)
                x_real_next = Rconv(self.act(x_real.clone())) - Iconv(self.act(x_imag.clone()))
                x_imag_next = Iconv(self.act(x_real.clone())) + Rconv(self.act(x_imag.clone()))
                
                #var = conv(self.act(var.real) + 1j * self.act(var.imag))
                #var = conv(var)
            x = torch.complex(x_real_next, x_imag_next)
            cache.append(x)
        return cache

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out

class Mamba_LDA(nn.Module):
    def __init__(self, **kwargs):
        super(Mamba_LDA, self).__init__()
        cur_iter = kwargs['n_block']
        
        self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.thresh_s = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)
        
        self.cur_iter = cur_iter
        
        self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        channel_num = kwargs['channel_num']
        self.channel_num = channel_num
        
        self.ImgNet = Complex_Mamba_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )
        
    def set_PhaseNo(self, cur_iter):
        self.cur_iter = cur_iter
        
    def phase(self, x, k, phase, gamma, mask, return_g=False):
        '''
            computation for each phase
        '''
        alpha = torch.abs(self.alphas[phase])
        beta = torch.abs(self.betas[phase])
        
        # update x
        #Ax = projection.apply(x, self.options)
        Fx = torch.fft.fft2(x, norm="ortho")
        # Fx = data_consistency(Fx, k, mask)
        residual = Fx - k
        # residual_S_new = Ax - z
        # grad_fx = projection_t.apply(residual_S_new, self.options)
        grad = torch.fft.ifft2(residual, norm="ortho")

        #c = x - alpha * grad_fx
        c = x - alpha * grad
        cache_x = self.ImgNet(c)
        u = c - beta * self.ImgNet.gradient(cache_x, gamma)
        
        Fu = torch.fft.fft2(u, norm="ortho")
        Fu = data_consistency(Fu, k, mask)
        u = torch.fft.ifft2(Fu, norm="ortho")
        
        if return_g:
            return u, cache_x[-1]
        return u
    
    def forward(self, x, k, mask, return_g=None):
        x_list = []
        g_list = []
        for phase in range(self.cur_iter):
            if return_g:
                x, g = self.phase(x, k, phase, 0.9**phase, mask, return_g)
                g_list.append(g)
            else:
                x = self.phase(x, k, phase, 0.9**phase, mask)
            x_list.append(x)
            
        if return_g:
            return x_list, g_list
        return x_list