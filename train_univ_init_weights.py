import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch

import os
import torch.nn.functional as F

from utils.dataset import universal_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.model import Universal_LDA

n_phase = 15
n_epoch = 50

init_seeds()
anatomies = ['brain', 'knee']
query = 'prostate'

model = Universal_LDA(n_block=n_phase, anatomies=anatomies, channel_num=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
model.to(device)

# initialize weights for the model


mask = 'cartesian'
acc = 5
dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], 
                         acc=acc, mask=mask)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#brain_dataset = anatomy_data('data/brain/brain_singlecoil_train.mat', acc=5)
#brain_loader = DataLoader(brain_dataset, batch_size=batch_size, shuffle=True)

# we initialize the model with a linear combination of independent models
# load the pre-trained models
brain_params = torch.load(f'universal_LDA/brain/checkpoints_{acc}_sampling_{mask}/checkpoint.pth')
knee_params = torch.load(f'universal_LDA/knee/checkpoints_{acc}_sampling_{mask}/checkpoint.pth')

for i in range(4):
    model.ImgNet.Rconvs[i].weight.data = (brain_params[f'ImgNet.Rconvs.{i}.weight'] + knee_params[f'ImgNet.Rconvs.{i}.weight']) / 2
    model.ImgNet.Rconvs[i].bias.data = (brain_params[f'ImgNet.Rconvs.{i}.bias'] + knee_params[f'ImgNet.Rconvs.{i}.bias']) / 2
    
    model.ImgNet.RconvsT[i].weight.data = (brain_params[f'ImgNet.RconvsT.{i}.weight'] + knee_params[f'ImgNet.RconvsT.{i}.weight']) / 2
    model.ImgNet.RconvsT[i].bias.data = (brain_params[f'ImgNet.RconvsT.{i}.bias'] + knee_params[f'ImgNet.RconvsT.{i}.bias']) / 2
    
    model.ImgNet.Iconvs[i].weight.data = (brain_params[f'ImgNet.Iconvs.{i}.weight'] + knee_params[f'ImgNet.Iconvs.{i}.weight']) / 2
    model.ImgNet.Iconvs[i].bias.data = (brain_params[f'ImgNet.Iconvs.{i}.bias'] + knee_params[f'ImgNet.Iconvs.{i}.bias']) / 2
    
    model.ImgNet.IconvsT[i].weight.data = (brain_params[f'ImgNet.IconvsT.{i}.weight'] + knee_params[f'ImgNet.IconvsT.{i}.weight']) / 2
    model.ImgNet.IconvsT[i].bias.data = (brain_params[f'ImgNet.IconvsT.{i}.bias'] + knee_params[f'ImgNet.IconvsT.{i}.bias']) / 2
    
model.h_dict['brain'].Rconv.weight.data = brain_params[f'ImgNet.Rconvs.1.weight']
model.h_dict['brain'].Rconv.bias.data = brain_params[f'ImgNet.Rconvs.1.bias']
model.h_dict['brain'].Iconv.weight.data = brain_params[f'ImgNet.Iconvs.1.weight']
model.h_dict['brain'].Iconv.bias.data = brain_params[f'ImgNet.Iconvs.1.bias']

model.h_dict['brain'].RconvT.weight.data = brain_params[f'ImgNet.RconvsT.1.weight']
model.h_dict['brain'].RconvT.bias.data = brain_params[f'ImgNet.RconvsT.1.bias']
model.h_dict['brain'].IconvT.weight.data = brain_params[f'ImgNet.IconvsT.1.weight']
model.h_dict['brain'].IconvT.bias.data = brain_params[f'ImgNet.IconvsT.1.bias']

model.h_dict['knee'].Rconv.weight.data = knee_params[f'ImgNet.Rconvs.1.weight']
model.h_dict['knee'].Rconv.bias.data = knee_params[f'ImgNet.Rconvs.1.bias']
model.h_dict['knee'].Iconv.weight.data = knee_params[f'ImgNet.Iconvs.1.weight']
model.h_dict['knee'].Iconv.bias.data = knee_params[f'ImgNet.Iconvs.1.bias']

model.h_dict['knee'].RconvT.weight.data = knee_params[f'ImgNet.RconvsT.1.weight']
model.h_dict['knee'].RconvT.bias.data = knee_params[f'ImgNet.RconvsT.1.bias']
model.h_dict['knee'].IconvT.weight.data = knee_params[f'ImgNet.IconvsT.1.weight']
model.h_dict['knee'].IconvT.bias.data = knee_params[f'ImgNet.IconvsT.1.bias']

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)


start_epoch = 1
start_phase = 3

save_dir = f"universal_LDA/universal_init_weights/checkpoints_{acc}_sampling_{mask}_start_{start_phase}"

#save_dir = f"universal_LDA/universal_init_weights/checkpoints_{acc}_sampling_{mask}_start_{start_phase}"

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    start_phase = checkpoint['phase']
    
    scheduler.step()
    scheduler.step()
    scheduler.step()
    print('Model loaded from checkpoint')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for PhaseNo in range(start_phase, n_phase+1, 2):
    model.set_PhaseNo(PhaseNo)
    PSNR_list = []
    loss_list = []
    
    for epoch_i in range(start_epoch, n_epoch+1):
        for i, data in enumerate(loader):
            # undersampled image, k-space, mask, original image, original k-space
            im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
            # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
            
            im_und = im_und.to(device)
            k_und = k_und.to(device)
            mask = mask.to(device)
            img_gnd = img_gnd.to(device)
            k_gnd = k_gnd.to(device)
            
            # forward pass
            optim.zero_grad()
            output = model(im_und, k_und, mask, anatomy[0])
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
            checkpoint = {
                                'phase': PhaseNo,
                                'epoch': epoch_i, 
                                'state_dict': model.state_dict(),
                                'optimizer': optim.state_dict(),
                                'scheduler': scheduler.state_dict()
                             }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint.pth'))
    start_epoch = 1
    scheduler.step()
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{PhaseNo}.pth'))