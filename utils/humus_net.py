import torch
import torch.nn as nn
from utils.humus_block import HUMUSBlock
from typing import Tuple

class VarNetBlock(nn.Module):    
    def __init__(self, model):
        super(VarNetBlock, self).__init__()
        self.model = model
        
        self.dc_weight = nn.Parameter(torch.ones(1))
        
    @staticmethod
    def complex_to_chan_dim(x) -> torch.Tensor:
        return torch.cat((x.real, x.imag), dim=1)
    
    @staticmethod
    def chan_complex_to_complex(x) -> torch.Tensor:
        return x[:, :x.shape[1]//2] + 1j*x[:, x.shape[1]//2:]
    
    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape # 1, 3 * 2, h, w
        #x = x.view(b, 2, c // 2 * h * w)
        x = x.view(b, c, h * w) # 1, 6, h * w

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std
    
    @staticmethod
    def unnorm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    def apply_model(self, x):
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x = self.model(x)
        x = self.unnorm(x, mean, std)
        
        x = self.chan_complex_to_complex(x)
        return x
    
    def forward(self, 
                current_kspace, 
                ref_kspace,
                mask):
        zero = torch.zeros_like(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        x = self.apply_model(torch.fft.ifft2(current_kspace, norm='ortho'))
        x = torch.fft.fft2(x, norm='ortho')
        x = current_kspace - soft_dc - x
        return x
    
class HUMUSNet(nn.Module):
    
    def __init__(self, num_cascades: int=4):
        super(HUMUSNet, self).__init__()
        self.num_cascades = num_cascades
        self.cascades = nn.ModuleList([
            VarNetBlock(HUMUSBlock(img_size=(256,256), in_chans=2)) for _ in range(num_cascades)
        ])
        
    def forward(self, masked_kspace, mask):
        mask = mask.bool()
        kspace_pred = masked_kspace.clone()
        
        for i, cascade in enumerate(self.cascades):
            kspace_pred = cascade(kspace_pred, masked_kspace, mask)
        
        out = torch.fft.ifft2(kspace_pred, norm='ortho')
        
        return out