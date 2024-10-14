import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch

import os
import torch.nn.functional as F

from utils.dataset import anatomy_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.model import LDA

n_phase = 15
n_epoch = 50

init_seeds()
anatomy = 'knee'
mask = 'cartesian'
model = LDA(n_block=n_phase, channel_num=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 2
model.to(device)

acc = 10

anatomy_dataset = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=acc, n=400, mask=mask)
anatomy_loader = DataLoader(anatomy_dataset, batch_size=batch_size, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.7)
if not mask:
    save_dir = f"universal_LDA/{anatomy}/checkpoints_{acc}_sampling_cartesian"
else:
    save_dir = f"universal_LDA/{anatomy}/checkpoints_{acc}_sampling_{mask}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for PhaseNo in range(3, n_phase+1, 2):
    model.set_PhaseNo(PhaseNo)
    PSNR_list = []
    loss_list = []
    
    for epoch_i in range(1, n_epoch+1):
        for i, data in enumerate(anatomy_loader):
            # undersampled image, k-space, mask, original image, original k-space
            im_und, k_und, mask, img_gnd, k_gnd = data
            # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
            
            im_und = im_und.to(device)
            k_und = k_und.to(device)
            mask = mask.to(device)
            img_gnd = img_gnd.to(device)
            k_gnd = k_gnd.to(device)    
            
            # forward pass
            optim.zero_grad()
            output = model(im_und, k_und, mask)
            output = torch.abs(output[-1]).clamp(0, 1)
            img_gnd = torch.abs(img_gnd)
            
            loss = torch.sum(torch.square(output - img_gnd))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
            loss.backward()
            optim.step()
            
            loss_list.append(loss.item())
        
            for j in range(batch_size):
                PSNR_list.append(psnr(np.abs(output[j].squeeze().cpu().detach().numpy()), img_gnd[j].squeeze().cpu().detach().numpy(), data_range=1))
            if (i+1) % 100 == 0:
                print(i+1, loss.item())
        avg_l = np.mean(loss_list)
        avg_p = np.mean(PSNR_list)
        epoch_data = '[Phase %02d] [Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
            (PhaseNo, epoch_i, n_epoch, avg_l, avg_p)
        print(epoch_data)
        
        if epoch_i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint.pth'))
    scheduler.step()