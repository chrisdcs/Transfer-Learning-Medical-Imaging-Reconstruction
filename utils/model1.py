
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
    

from functools import partial
import math
import time
import torch
import warnings


WITH_SELECTIVESCAN_OFLEX = True
WITH_SELECTIVESCAN_CORE = False
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
    warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
    print("Can not import selective_scan_cuda_oflex. This affects speed.", flush=True)
try:
    import selective_scan_cuda_core
    print("selective_scan_cuda_core imported", flush=True)
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
try:
    import selective_scan_cuda
    print("selective_scan_cuda imported", flush=True)
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False


def selective_scan_torch(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True, 
    *args,
    **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (B, C, L)
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        backend = "oflex"# if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        #backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        #backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True,
    backend=None,
):
    WITH_CUDA = (WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA)
    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


# fvcore flops =======================================
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)



class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class SS2Dv0(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=256,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")
        
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -self.A_logs.float().exp() # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Complex_VMamba_Block(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Complex_VMamba_Block, self).__init__()
        
        n_feats = kwargs['n_feats']
        n_convs = kwargs['n_convs']
        k_size = kwargs['k_size']
        padding = kwargs['padding']
        self.padding = padding
        
        self.soft_thr = nn.Parameter(torch.Tensor([0.002]),requires_grad=True)
        
        #in_chans=1
        #embed_dim=96
        #patch_size=4
        
        #Rconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding),
        #          nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
        #          nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
        #         [MambaLayer(n_feats, n_feats, 4, 2)]
                 
        Rconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                 [SS2Dv0(16)]
        self.Rconvs = nn.ModuleList(Rconvs)
        
        #Iconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding),
        #          nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
        #          nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
        #         [MambaLayer(n_feats, n_feats, 4, 2)]
        
        Iconvs = [nn.Conv2d(1, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                  nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                 [SS2Dv0(16)]
                 #[nn.Conv2d(n_feats, n_feats, kernel_size=k_size, padding=padding) for i in range(n_convs-1)]
        self.Iconvs = nn.ModuleList(Iconvs)
        
        
        #RconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding),
        #           nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
        #           nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)]+\
        #          [MambaLayer(n_feats, n_feats, 4, 2)]
                   
        RconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                  [SS2Dv0(16)]
                  
        self.RconvsT = nn.ModuleList(RconvsT)
        
        #IconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding),
        #           nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
        #           nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)]+\
        #          [MambaLayer(n_feats, n_feats, 4, 2)]
        
        IconvsT = [nn.ConvTranspose2d(n_feats, 1, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding),
                   nn.ConvTranspose2d(n_feats, n_feats, kernel_size=k_size, padding=padding)] + \
                  [SS2Dv0(16)]
                  
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
            
            if i == len(forward_cache) - 1:
                out_real = out_real.permute(0, 2, 3, 1).contiguous()
                out_imag = out_imag.permute(0, 2, 3, 1).contiguous()
            out_real_next = self.RconvsT[i](out_real) - self.IconvsT[i](out_imag)
            out_img_next = self.IconvsT[i](out_real) + self.RconvsT[i](out_imag)
            
            if i == len(forward_cache) - 1:
                out_real_next = out_real_next.permute(0, 3, 1, 2).contiguous()
                out_img_next = out_img_next.permute(0, 3, 1, 2).contiguous()
            #out = out_real_next + 1j * out_img_next
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
            elif i == len(self.Rconvs) - 1:
                x_real = x_real.permute(0, 2, 3, 1).contiguous()
                x_imag = x_imag.permute(0, 2, 3, 1).contiguous()
                x_real_next = Rconv(self.act(x_real.clone()))
                x_imag_next = Iconv(self.act(x_imag.clone()))
                x_real_next = x_real_next.permute(0, 3, 1, 2).contiguous()
                x_imag_next = x_imag_next.permute(0, 3, 1, 2).contiguous()
            else:
                # x_real, x_imag = self.act(x_real), self.act(x_imag)
                x_real_next = Rconv(self.act(x_real.clone())) - Iconv(self.act(x_imag.clone()))
                x_imag_next = Iconv(self.act(x_real.clone())) + Rconv(self.act(x_imag.clone()))
                
                #var = conv(self.act(var.real) + 1j * self.act(var.imag))
                #var = conv(var)
            x = torch.complex(x_real_next, x_imag_next)
            cache.append(x)
        return cache
    
class VMamba_LDA(nn.Module):
    def __init__(self, **kwargs):
        super(VMamba_LDA, self).__init__()
        cur_iter = kwargs['n_block']
        
        self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.thresh_s = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)
        
        self.cur_iter = cur_iter
        
        self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        channel_num = kwargs['channel_num']
        self.channel_num = channel_num
        
        self.ImgNet = nn.ModuleList([Complex_VMamba_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )] * kwargs['n_block'])
    
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
        cache_x = self.ImgNet[phase](c)
        u = c - beta * self.ImgNet[phase].gradient(cache_x, gamma)
        
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