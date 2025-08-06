import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch

import os
import torch.nn.functional as F

from utils.dataset import *
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.model import Universal_LDA

n_phase = 15
n_epoch = 50

start_epoch = 1
start_phase = 3

mode = 'mode'

init_seeds()

if mode == 'anatomy':
    anatomies = ['brain', 'knee']
elif mode == 'sampling':
    anatomies = ['10', '5', '3']
elif mode == 'dataset':
    anatomies = ['imagenet']
elif mode == 'domain':
    anatomies = ['imagenet', 'cifar10']

model = Universal_LDA(n_block=n_phase, anatomies=anatomies, channel_num=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
model.to(device)

# initialize weights for the model


mask = 'cartesian'

if mode == 'anatomy':
    acc = 5
    dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], 
                            acc=acc, mask=mask)
    save_dir = f"universal_LDA/universal_init_weights/checkpoints_{acc}_sampling_{mask}_start_{start_phase}"
elif mode == 'sampling':
    anatomy = "brain"
    file = f'data/{anatomy}/brain_singlecoil_train.mat'
    dataset = universal_sampling_data(file, [10, 5, 3.33], "cartesian")
    #save_dir = f"universal_LDA/universal_init_weights/cross_sampling/checkpoints_{anatomy}_{mask}"
    save_dir = f"universal_LDA/universal/cross_sampling/checkpoints_{anatomy}_{mask}"
elif mode == 'dataset':
    acc = 5
    anatomy = 'imagenet'
    file = f'data/{anatomy}/{anatomy}_singlecoil_train.mat'
    #files = [f'data/{anatomy}/{anatomy}_singlecoil_train.mat' for anatomy in anatomies]
    dataset = universal_data([file], acc=acc, mask = "cartesian", n=400)
    #dataset = universal_data(files, acc=acc, mask = "cartesian", n=400)
    save_dir = f"universal_LDA/universal_init_weights/cross_dataset/checkpoints_{acc}_{mask}"
    #save_dir = f"universal_LDA/universal/cross_dataset/checkpoints_{acc}_{mask}"
elif mode == 'domain':
    acc = 5
    files = [f'data/{anatomy}/{anatomy}_singlecoil_train.mat' for anatomy in anatomies]
    dataset = universal_data(files, acc=acc, mask=mask, n=400)
    save_dir = f"universal_LDA/universal_init_weights/cross_domain/checkpoints_{acc}_{mask}"

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#brain_dataset = anatomy_data('data/brain/brain_singlecoil_train.mat', acc=5)
#brain_loader = DataLoader(brain_dataset, batch_size=batch_size, shuffle=True)

# we initialize the model with a linear combination of independent models
# load the pre-trained models
#"""
if mode == 'anatomy':
    #brain_params = torch.load(f'universal_LDA/brain/checkpoints_{acc}_sampling_{mask}/checkpoint.pth')
    #knee_params = torch.load(f'universal_LDA/knee/checkpoints_{acc}_sampling_{mask}/checkpoint.pth')

    params = {}
    for name in anatomies:
        params[name] = torch.load(f'universal_LDA/{name}/checkpoints_{acc}_sampling_{mask}/checkpoint.pth')
elif mode == 'sampling':
    params = {}
    sampling_rate = {'10': '10', '5': '5', '3': '3.33'}
    for name in anatomies:
        params[name] = torch.load(f'universal_LDA/brain/cross_sampling/checkpoints_{sampling_rate[name]}_{mask}_phase_15_samples_300/checkpoint.pth')
elif mode == 'dataset':
    # for cross-dataset transfer learning
    params = {}
    for name in anatomies:
        params[name] = torch.load(f'universal_LDA/{name}/checkpoints_{acc}_sampling_{mask}_phase_15_samples_400/checkpoint.pth')


for i in range(4):
    # initialize the weights of universal g with the average of the weights of the independent models
    model.ImgNet.Rconvs[i].weight.data = torch.mean(torch.stack([params[name][f'ImgNet.Rconvs.{i}.weight'] for name in anatomies]), dim=0)
    model.ImgNet.Rconvs[i].bias.data = torch.mean(torch.stack([params[name][f'ImgNet.Rconvs.{i}.bias'] for name in anatomies]), dim=0)
    
    model.ImgNet.RconvsT[i].weight.data = torch.mean(torch.stack([params[name][f'ImgNet.RconvsT.{i}.weight'] for name in anatomies]), dim=0)
    model.ImgNet.RconvsT[i].bias.data = torch.mean(torch.stack([params[name][f'ImgNet.RconvsT.{i}.bias'] for name in anatomies]), dim=0)
    
    model.ImgNet.Iconvs[i].weight.data = torch.mean(torch.stack([params[name][f'ImgNet.Iconvs.{i}.weight'] for name in anatomies]), dim=0)
    model.ImgNet.Iconvs[i].bias.data = torch.mean(torch.stack([params[name][f'ImgNet.Iconvs.{i}.bias'] for name in anatomies]), dim=0)
    
    model.ImgNet.IconvsT[i].weight.data = torch.mean(torch.stack([params[name][f'ImgNet.IconvsT.{i}.weight'] for name in anatomies]), dim=0)
    model.ImgNet.IconvsT[i].bias.data = torch.mean(torch.stack([params[name][f'ImgNet.IconvsT.{i}.bias'] for name in anatomies]), dim=0)
    
for name in anatomies:
    # initialize the weights of the anatomy-specific h with the weights of the independent models shallow layer
    model.h_dict[name].Rconv.weight.data = params[name][f'ImgNet.Rconvs.1.weight']
    model.h_dict[name].Rconv.bias.data = params[name][f'ImgNet.Rconvs.1.bias']
    model.h_dict[name].Iconv.weight.data = params[name][f'ImgNet.Iconvs.1.weight']
    model.h_dict[name].Iconv.bias.data = params[name][f'ImgNet.Iconvs.1.bias']
    
    model.h_dict[name].RconvT.weight.data = params[name][f'ImgNet.RconvsT.1.weight']
    model.h_dict[name].RconvT.bias.data = params[name][f'ImgNet.RconvsT.1.bias']
    model.h_dict[name].IconvT.weight.data = params[name][f'ImgNet.IconvsT.1.weight']
    model.h_dict[name].IconvT.bias.data = params[name][f'ImgNet.IconvsT.1.bias']
#"""
    
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)

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
            
            if torch.is_tensor(anatomy[0]):
                if anatomy[0].item() == 3.33:
                    anatomy = ['3']
                else:
                    anatomy = [str(anatomy[0].item())]
            
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