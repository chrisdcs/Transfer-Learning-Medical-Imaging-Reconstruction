import torch
import torch.nn as nn
import torch.nn.functional as F

class sigma_activation(nn.Module):
    def __init__(self, ddelta):
        super(sigma_activation, self).__init__()
        self.relu = nn.ReLU(inplace=True)      
        self.ddelta = ddelta
        self.coeff = 1.0 / (4.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu = self.relu(x_i)
        x_square = torch.square(x_i) * self.coeff
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu, x_square + 0.5*x_i + 0.25 * self.ddelta)
    
class sigma_derivative(nn.Module):
    def __init__(self, ddelta):
        super(sigma_derivative, self).__init__()
        self.ddelta = ddelta
        self.coeff2 = 1.0 / (2.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu_deri = torch.where(x_i > 0, torch.ones_like(x_i), torch.zeros_like(x_i))
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu_deri, self.coeff2 *x_i + 0.5)

class Complex_Learnable_Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Complex_Learnable_Block, self).__init__()
        
        n_feats = kwargs['n_feats']
        n_convs = kwargs['n_convs']
        k_size = kwargs['k_size']
        padding = kwargs['padding']
        self.padding = padding
        
        self.soft_thr = nn.Parameter(torch.Tensor([0.002]),requires_grad=True)
        Rconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding)] + \
                 [nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
        
        self.Rconvs = nn.ModuleList(Rconvs)
        
        Iconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding)] + \
                 [nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
        self.Iconvs = nn.ModuleList(Iconvs)
        
        
        RconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding)] + \
                  [nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
                  
        self.RconvsT = nn.ModuleList(RconvsT)
        
        IconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding)] + \
                  [nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
                  
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
            tmp_real, tmp_imag = self.act_der(forward_cache[i-1].clone().real), self.act_der(forward_cache[i-1].clone().imag)
            #out_real_next = F.conv_transpose2d(out_real, self.Rconvs[i].weight, padding=self.padding) - F.conv_transpose2d(out_imag, self.Iconvs[i].weight, padding=self.padding)
            #out_img_next = F.conv_transpose2d(out_real, self.Iconvs[i].weight, padding=self.padding) + F.conv_transpose2d(out_imag, self.Rconvs[i].weight, padding=self.padding)
            out_real_next = self.RconvsT[i](out_real) - self.IconvsT[i](out_imag)
            out_img_next = self.IconvsT[i](out_real) + self.RconvsT[i](out_imag)

            out = out_real_next * tmp_real - out_img_next * tmp_imag + 1j * (out_real_next * tmp_imag + out_img_next * tmp_real)
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


# Exact Learnable Block
class Learnable_Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Learnable_Block, self).__init__()
        
        n_feats = kwargs['n_feats']
        n_convs = kwargs['n_convs']
        k_size = kwargs['k_size']
        padding = kwargs['padding']
        self.padding = padding
        
        self.soft_thr = nn.Parameter(torch.Tensor([0.002]),requires_grad=True)
        convs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding).to(torch.complex64)] + \
                [nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding).to(torch.complex64) for i in range(n_convs-1)]
        self.convs = nn.ModuleList(convs)
        
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
            tmp = self.act_der(forward_cache[i-1].clone().real) + 1.j * self.act_der(forward_cache[i-1].clone().imag)
            out = F.conv_transpose2d(out, self.convs[i].weight, padding=self.padding)*tmp# * (self.act_der(forward_cache[i-1].real) + 1j * self.act_der(forward_cache[i-1].imag))
        out = F.conv_transpose2d(out, self.convs[0].weight, padding=self.padding)
        return out
    
    def smoothed_reg(self, forward_cache, gamma):
        soft_thr = torch.abs(self.soft_thr) * gamma
        g = forward_cache[-1]
        
        norm_g = torch.norm(g, dim = 1, keepdim=True)
        reg = torch.where(norm_g > soft_thr, norm_g - torch.div(soft_thr,2), 
                          torch.square(norm_g)/(torch.mul(soft_thr,2)))
        reg = torch.flatten(reg, start_dim=1)
        reg = torch.sum(reg, -1, keepdim=True)
        
        return reg
        
    def forward(self, var):
        cache = []
        for i, conv in enumerate(self.convs):
                if i == 0:
                    var = conv(var)
                else:
                    tmp = self.act(var.clone().real) + 1j * self.act(var.clone().imag)
                    var = conv(tmp)
                    #var = conv(self.act(var.real) + 1j * self.act(var.imag))
                    #var = conv(var)
                cache.append(var)
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

class LDA(nn.Module):
    def __init__(self, **kwargs):
        super(LDA, self).__init__()
        cur_iter = kwargs['n_block']
        
        self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.thresh_s = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)
        
        self.cur_iter = cur_iter
        
        self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        channel_num = kwargs['channel_num']
        self.channel_num = channel_num
        
        self.ImgNet = Complex_Learnable_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )
        
    def set_PhaseNo(self, cur_iter):
        self.cur_iter = cur_iter
        
    def phase(self, x, k, phase, gamma, mask):
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
        
        return u
    
    def forward(self, x, k, mask):
        x_list = []
        for phase in range(self.cur_iter):
            x = self.phase(x, k, phase, 0.9**phase, mask)
            x_list.append(x)
            
        return x_list


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        real_part = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag_part = self.real_conv(x.imag) + self.imag_conv(x.real)
        return torch.complex(real_part, imag_part)

class Init_CNN(nn.Module):
    def __init__(self):
        super(Init_CNN, self).__init__()
        self.conv1 = ComplexConv2d(1, 16, 3)
        self.conv2 = ComplexConv2d(16, 16, 3)
        self.conv3 = ComplexConv2d(16, 16, 3)
        self.conv4 = ComplexConv2d(16, 1, 3)
        
    def forward(self, x, mask):
        x_inp = x.clone()
        x = self.conv1(x)
        x = torch.complex(F.relu(x.real), F.relu(x.imag))
        x = self.conv2(x)
        x = torch.complex(F.relu(x.real), F.relu(x.imag))
        x = self.conv3(x)
        x = torch.complex(F.relu(x.real), F.relu(x.imag))
        x = self.conv4(x)
        
        x = data_consistency(x, x_inp, mask)
        
        return x
        
def complex_mse_loss(input, target):
    # Assuming input and target are complex tensors with real and imaginary parts
    real_diff = input.real - target.real
    imag_diff = input.imag - target.imag
    mse_loss = torch.mean(0.5 * (real_diff**2 + imag_diff**2))
    return mse_loss

class LDA_vis(nn.Module):
    def __init__(self, **kwargs):
        super(LDA_vis, self).__init__()
        cur_iter = kwargs['n_block']
        
        self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.thresh_s = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)
        
        self.cur_iter = cur_iter
        
        self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        channel_num =16
        self.channel_num = channel_num
        
        self.ImgNet = Complex_Learnable_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )
        
    def set_PhaseNo(self, cur_iter):
        self.cur_iter = cur_iter
        
    def phase(self, x, k, phase, gamma, mask):
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
        
        return u, cache_x
    
    def forward(self, x, k, mask):
        x_list = []
        g_list = []
        for phase in range(self.cur_iter):
            x, g = self.phase(x, k, phase, 0.9**phase, mask)
            x_list.append(x)
            g_list.append(g)
            
        return x_list, g_list

class Domain_Transform(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Domain_Transform, self).__init__()
        # lets try a simple one layer conv
        n_feats = kwargs['n_feats']
        padding = kwargs['padding']
        k_size = kwargs['k_size']
        self.Rconv = nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)
        self.Iconv = nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)
        
        self.RconvT = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)
        self.IconvT = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)
    
    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        x_real_out = self.Rconv(x_real) - self.Iconv(x_imag)
        x_imag_out = self.Rconv(x_imag) + self.Iconv(x_real)
        
        out = torch.complex(x_real_out, x_imag_out)
        
        return out

class Universal_LDA(nn.Module):
    def __init__(self, **kwargs):
        super(Universal_LDA, self).__init__()
        anatomies = kwargs['anatomies']
        channel_num = kwargs['channel_num']
        # channel_num = 32
        self.h_dict = nn.ModuleDict(
            {
                anatomy: 
                Domain_Transform(
                    n_feats=channel_num,
                    k_size=3,
                    padding=1,
                    ) for anatomy in anatomies
            }
        )
        
        cur_iter = kwargs['n_block']
        
        # self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.soft_thr = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.Tensor([0.002]), requires_grad=True) for anatomy in anatomies
                        })
        
        self.cur_iter = cur_iter
        
        # self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        # self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        self.alphas = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True) for anatomy in anatomies
                        })
        self.betas = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True) for anatomy in anatomies
                        })
        
        # complex learnable blocks are still the same
        # except the gradient part
        # customize gradient function under universal LDA
        self.ImgNet = Complex_Learnable_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )
    
    def set_PhaseNo(self, cur_iter):
        self.cur_iter = cur_iter
    
    def gradient(self, forward_cache, gamma, anatomy):
        soft_thr = torch.abs(self.soft_thr[anatomy]) * gamma
        hg = forward_cache.pop()
        
        # compute gradient of smoothed regularization
        norm_hg = torch.norm(hg, dim = 1, keepdim=True)
        denominator = torch.where(norm_hg > soft_thr, norm_hg, soft_thr)
        out = torch.div(hg, denominator)
        out_real, out_imag = out.real, out.imag
        out_real_next = self.h_dict[anatomy].RconvT(out_real) - self.h_dict[anatomy].IconvT(out_imag)
        out_img_next = self.h_dict[anatomy].IconvT(out_real) + self.h_dict[anatomy].RconvT(out_imag)
        out = out_real_next + 1j * out_img_next
        
        #out_real_next = F.conv_transpose2d(out_real, self.h_dict[anatomy][phase].Rconv.weight, padding=self.ImgNet.padding) - \
        #                F.conv_transpose2d(out_imag, self.h_dict[anatomy][phase].Iconv.weight, padding=self.ImgNet.padding)
        #out_img_next = F.conv_transpose2d(out_real, self.h_dict[anatomy][phase].Iconv.weight, padding=self.ImgNet.padding) + \
        #                F.conv_transpose2d(out_imag, self.h_dict[anatomy][phase].Rconv.weight, padding=self.ImgNet.padding)
        #out = out_real_next + 1j * out_img_next
        
        for i in range(len(forward_cache)-1, 0, -1):
            out_real, out_imag = out.real, out.imag
            # tmp = self.act_der(forward_cache[i].clone().real) + 1.j * self.act_der(forward_cache[i].clone().imag)
            tmp_real, tmp_imag = self.ImgNet.act_der(forward_cache[i-1].clone().real), self.ImgNet.act_der(forward_cache[i-1].clone().imag)
            #out_real_next = F.conv_transpose2d(out_real, self.ImgNet.Rconvs[i].weight, padding=self.ImgNet.padding) - \
            #                F.conv_transpose2d(out_imag, self.ImgNet.Iconvs[i].weight, padding=self.ImgNet.padding)
            #out_img_next = F.conv_transpose2d(out_real, self.ImgNet.Iconvs[i].weight, padding=self.ImgNet.padding) + \
            #                F.conv_transpose2d(out_imag, self.ImgNet.Rconvs[i].weight, padding=self.ImgNet.padding)
            out_real_next = self.ImgNet.RconvsT[i](out_real) - self.ImgNet.IconvsT[i](out_imag)
            out_img_next = self.ImgNet.IconvsT[i](out_real) + self.ImgNet.RconvsT[i](out_imag)
            out = out_real_next * tmp_real - out_img_next * tmp_imag + 1j * (out_real_next * tmp_imag + out_img_next * tmp_real)
        
        out_real, out_imag = out.real, out.imag
        #out_real_next = F.conv_transpose2d(out_real, self.ImgNet.Rconvs[0].weight, padding=self.ImgNet.padding) - \
        #                F.conv_transpose2d(out_imag, self.ImgNet.Iconvs[0].weight, padding=self.ImgNet.padding)
        #out_img_next = F.conv_transpose2d(out_real, self.ImgNet.Iconvs[0].weight, padding=self.ImgNet.padding) + \
        #                F.conv_transpose2d(out_imag, self.ImgNet.Rconvs[0].weight, padding=self.ImgNet.padding)
        out_real_next = self.ImgNet.RconvsT[0](out_real) - self.ImgNet.IconvsT[0](out_imag)
        out_img_next = self.ImgNet.IconvsT[0](out_real) + self.ImgNet.RconvsT[0](out_imag)
        out = out_real_next + 1j * out_img_next
        
        return out
        
    
    def phase(self,x, k, phase, gamma, mask, anatomy):
        '''
            computation for each phase
        '''
        alpha = torch.abs(self.alphas[anatomy][phase])
        beta = torch.abs(self.betas[anatomy][phase])
        
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
        cache_univ = self.ImgNet(c)
        # g = cache_univ.pop()
        
        # pass to domain transform
        hg = self.h_dict[anatomy](cache_univ[-1])
        cache_univ.append(hg)
        
        # calculate gradient
        u = c - beta * self.gradient(cache_univ, gamma, anatomy)
        
        Fu = torch.fft.fft2(u, norm="ortho")
        Fu = data_consistency(Fu, k, mask)
        u = torch.fft.ifft2(Fu, norm="ortho")
        
        return u
    
    def forward(self, x, k, mask, anatomy):
        x_list = []
        for phase in range(self.cur_iter):
            x = self.phase(x, k, phase, 0.9**phase, mask, anatomy)
            x_list.append(x)
            
        return x_list
    
    
class Universal_LDA_vis(nn.Module):
    def __init__(self, **kwargs):
        super(Universal_LDA_vis, self).__init__()
        anatomies = kwargs['anatomies']
        
        channel_num = 16
        self.h_dict = nn.ModuleDict(
            {
                anatomy: 
                Domain_Transform(
                    n_feats=channel_num,
                    k_size=3,
                    padding=1,
                    ) for anatomy in anatomies
            }
        )
        
        cur_iter = kwargs['n_block']
        
        # self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.soft_thr = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.Tensor([0.002]), requires_grad=True) for anatomy in anatomies
                        })
        
        self.cur_iter = cur_iter
        
        # self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        # self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        self.alphas = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True) for anatomy in anatomies
                        })
        self.betas = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True) for anatomy in anatomies
                        })
        
        # complex learnable blocks are still the same
        # except the gradient part
        # customize gradient function under universal LDA
        self.ImgNet = Complex_Learnable_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )
    
    def set_PhaseNo(self, cur_iter):
        self.cur_iter = cur_iter
    
    def gradient(self, forward_cache, gamma, anatomy):
        soft_thr = torch.abs(self.soft_thr[anatomy]) * gamma
        hg = forward_cache.pop()
        
        # compute gradient of smoothed regularization
        norm_hg = torch.norm(hg, dim = 1, keepdim=True)
        denominator = torch.where(norm_hg > soft_thr, norm_hg, soft_thr)
        out = torch.div(hg, denominator)
        out_real, out_imag = out.real, out.imag
        out_real_next = self.h_dict[anatomy].RconvT(out_real) - self.h_dict[anatomy].IconvT(out_imag)
        out_img_next = self.h_dict[anatomy].IconvT(out_real) + self.h_dict[anatomy].RconvT(out_imag)
        out = out_real_next + 1j * out_img_next
        
        #out_real_next = F.conv_transpose2d(out_real, self.h_dict[anatomy][phase].Rconv.weight, padding=self.ImgNet.padding) - \
        #                F.conv_transpose2d(out_imag, self.h_dict[anatomy][phase].Iconv.weight, padding=self.ImgNet.padding)
        #out_img_next = F.conv_transpose2d(out_real, self.h_dict[anatomy][phase].Iconv.weight, padding=self.ImgNet.padding) + \
        #                F.conv_transpose2d(out_imag, self.h_dict[anatomy][phase].Rconv.weight, padding=self.ImgNet.padding)
        #out = out_real_next + 1j * out_img_next
        
        for i in range(len(forward_cache)-1, 0, -1):
            out_real, out_imag = out.real, out.imag
            # tmp = self.act_der(forward_cache[i].clone().real) + 1.j * self.act_der(forward_cache[i].clone().imag)
            tmp_real, tmp_imag = self.ImgNet.act_der(forward_cache[i-1].clone().real), self.ImgNet.act_der(forward_cache[i-1].clone().imag)
            #out_real_next = F.conv_transpose2d(out_real, self.ImgNet.Rconvs[i].weight, padding=self.ImgNet.padding) - \
            #                F.conv_transpose2d(out_imag, self.ImgNet.Iconvs[i].weight, padding=self.ImgNet.padding)
            #out_img_next = F.conv_transpose2d(out_real, self.ImgNet.Iconvs[i].weight, padding=self.ImgNet.padding) + \
            #                F.conv_transpose2d(out_imag, self.ImgNet.Rconvs[i].weight, padding=self.ImgNet.padding)
            out_real_next = self.ImgNet.RconvsT[i](out_real) - self.ImgNet.IconvsT[i](out_imag)
            out_img_next = self.ImgNet.IconvsT[i](out_real) + self.ImgNet.RconvsT[i](out_imag)
            out = out_real_next * tmp_real - out_img_next * tmp_imag + 1j * (out_real_next * tmp_imag + out_img_next * tmp_real)
        
        out_real, out_imag = out.real, out.imag
        #out_real_next = F.conv_transpose2d(out_real, self.ImgNet.Rconvs[0].weight, padding=self.ImgNet.padding) - \
        #                F.conv_transpose2d(out_imag, self.ImgNet.Iconvs[0].weight, padding=self.ImgNet.padding)
        #out_img_next = F.conv_transpose2d(out_real, self.ImgNet.Iconvs[0].weight, padding=self.ImgNet.padding) + \
        #                F.conv_transpose2d(out_imag, self.ImgNet.Rconvs[0].weight, padding=self.ImgNet.padding)
        out_real_next = self.ImgNet.RconvsT[0](out_real) - self.ImgNet.IconvsT[0](out_imag)
        out_img_next = self.ImgNet.IconvsT[0](out_real) + self.ImgNet.RconvsT[0](out_imag)
        out = out_real_next + 1j * out_img_next
        
        return out
        
    
    def phase(self,x, k, phase, gamma, mask, anatomy):
        '''
            computation for each phase
        '''
        alpha = torch.abs(self.alphas[anatomy][phase])
        beta = torch.abs(self.betas[anatomy][phase])
        
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
        cache_univ = self.ImgNet(c)
        # g = cache_univ.pop()
        
        # pass to domain transform
        hg = self.h_dict[anatomy](cache_univ[-1])
        cache_univ.append(hg)
        
        # calculate gradient
        u = c - beta * self.gradient(cache_univ, gamma, anatomy)
        
        Fu = torch.fft.fft2(u, norm="ortho")
        Fu = data_consistency(Fu, k, mask)
        u = torch.fft.ifft2(Fu, norm="ortho")
        
        return u, cache_univ, hg
    
    def forward(self, x, k, mask, anatomy):
        x_list = []
        g_list = []
        hg_list = []
        for phase in range(self.cur_iter):
            x, g, hg = self.phase(x, k, phase, 0.9**phase, mask, anatomy)
            x_list.append(x)
            g_list.append(g)
            hg_list.append(hg)
            
        return x_list, g_list, hg_list