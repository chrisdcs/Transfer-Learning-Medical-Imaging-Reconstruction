import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch

import os
import torch.nn.functional as F

from utils.dataset import universal_data, anatomy_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.model import Universal_LDA

mode = "sampling"
n_phase = 15
n_epoch = 100
mask = 'cartesian'

init_seeds()
if mode == "anatomy":
    anatomies = ['brain', 'knee']#['brain', 'knee', 'cardiac']
    anatomy = ['prostate']
elif mode == "sampling":
    anatomies = ['10', '5', '3']
    anatomy = ['6']
elif mode == "dataset":
    anatomies = ['imagenet'] # for cross-dataset transfer learning, you can use 'imagenet' or any other dataset you have
    anatomy = ['fastMRI']


model = Universal_LDA(n_block=n_phase, anatomies=anatomies, channel_num=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 2

n_samples = 100
model.add_anatomy(anatomy[0], 16)
model.to(device)
if mode == "anatomy":
    acc = 5
    #model.load_state_dict(torch.load(f'universal_LDA/universal_init_weights/checkpoints_{acc}_sampling_{mask}_start_3/checkpoint.pth')['state_dict'],
    #                    strict=False)
    model.load_state_dict(torch.load(f'universal_LDA/{anatomy[0]}/checkpoints_transfer_10_sampling_{mask}_samples_100/checkpoint.pth')['state_dict'],
                          strict=False)
elif mode == 'sampling':
    # for split initialization training, or for half split training
    #model.load_state_dict(torch.load(f'universal_LDA/universal_init_weights/cross_sampling/checkpoints_brain_{mask}/checkpoint.pth')['state_dict'],
    #                    strict=False)
    #model.load_state_dict(torch.load(f'universal_LDA/brain/checkpoints_transfer_13.33_sampling_{mask}_samples_{n_samples}/checkpoint.pth')['state_dict'],
    #                    strict=False)
    model.load_state_dict(torch.load(f'universal_LDA/universal/cross_sampling/checkpoints_brain_{mask}/checkpoint.pth')['state_dict'],
                        strict=False)
elif mode == 'dataset':
    #model.load_state_dict(torch.load('universal_LDA/universal_init_weights/cross_dataset/checkpoints_5_cartesian/checkpoint.pth')['state_dict'],
    #                      strict=False)
    #model.load_state_dict(torch.load(f'universal_LDA/{anatomy[0]}/checkpoints_transfer_10_cartesian_samples_{n_samples}/checkpoint.pth')['state_dict'],
    #                    )
    model.load_state_dict(torch.load('universal_LDA/universal/cross_dataset/checkpoints_5_cartesian/checkpoint.pth')['state_dict'],
                          strict=False)

# freeze the weights of the pretrained anatomy-specific layers
for name, param in model.named_parameters():
    if any(ana in name for ana in anatomies) or 'ImgNet' in name:
        # freeze all parameters of the anatomy-specific layers
        param.requires_grad = False
# verify if the anatomy-specific layers are frozen
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

#dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], acc=5)
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#acc = 10
if mode == "anatomy":
    acc = 5
    transfer_dataset = anatomy_data(f'data/{anatomy[0]}/{anatomy[0]}_singlecoil_train.mat', acc=acc, n=n_samples, mask=mask)
    print(f"number of samples in {anatomy[0]} dataset: ", len(transfer_dataset))
    transfer_loader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True)
elif mode == 'sampling':
    acc = 6.66
    transfer_dataset = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=acc, n=n_samples, mask=mask)
    transfer_loader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True)
elif mode == 'dataset':
    acc = 5
    transfer_dataset = anatomy_data(f'data/{anatomy[0]}/{anatomy[0]}_singlecoil_train.mat', acc=acc, n=n_samples, mask=mask)
    transfer_loader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
if mode == "anatomy":
    #save_dir = f"universal_LDA/{anatomy[0]}/checkpoints_transfer_{acc}_sampling_{mask}_samples_{n_samples}"
    save_dir = f"universal_LDA/{anatomy[0]}/checkpoints_transfer_{acc}_sampling_{mask}_samples_{n_samples}_split_init_training"#"
elif mode == 'sampling':
    # use half of k-space for split initialization training
    #save_dir = f"universal_LDA/brain/checkpoints_transfer_{acc}_sampling_{mask}_samples_{n_samples}" # for brain
    #save_dir = f"universal_LDA/brain/checkpoints_transfer_{acc}_sampling_{mask}_samples_{n_samples}_split_init_training"
    save_dir = f"universal_LDA/brain/checkpoints_transfer_{acc}_sampling_{mask}_samples_{n_samples}_simple"
elif mode == 'dataset':
    #save_dir = f"universal_LDA/{anatomy[0]}/checkpoints_transfer_{acc}_{mask}_samples_{n_samples}" # for cross-dataset transfer learning
    #save_dir = f"universal_LDA/{anatomy[0]}/checkpoints_transfer_{acc}_{mask}_samples_{n_samples}_split_init_training"
    save_dir = f"universal_LDA/{anatomy[0]}/checkpoints_transfer_{acc}_{mask}_samples_{n_samples}_simple"
start_phase = 3
start_epoch = 1

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

#anatomy = ['cardiac']
# n_phase = 9
for PhaseNo in range(start_phase, n_phase+1, 2):
    model.set_PhaseNo(PhaseNo)
    PSNR_list = []
    loss_list = []
    
    for epoch_i in range(start_epoch, n_epoch+1):
        for i, data in enumerate(transfer_loader):
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
            output = model(im_und, k_und, mask, anatomy[0])
            output = torch.abs(output[-1]).clamp(0, 1)
            img_gnd = torch.abs(img_gnd)
            
            loss = torch.sum(torch.square(output - img_gnd))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
            loss.backward()
            optim.step()
            
            loss_list.append(loss.item())
        
            for j in range(min(batch_size, output.shape[0])):
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