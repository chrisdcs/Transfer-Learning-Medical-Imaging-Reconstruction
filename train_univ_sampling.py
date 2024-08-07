import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch

import os
import torch.nn.functional as F

from utils.dataset import anatomy_data
from utils.dataset import universal_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.model import Universal_LDA

n_phase = 15
n_epoch = 100

init_seeds()
sampling_rates = ['4', '8', '12', '15']
anatomy = 'brain'

model = Universal_LDA(n_block=n_phase, anatomies=sampling_rates, channel_num=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
model.to(device)

dataset1 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=4, n=400)
loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)

dataset2 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=8, n=400)
loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

dataset3 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=12, n=400)
loader3 = DataLoader(dataset3, batch_size=batch_size, shuffle=True)

dataset4 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=15, n=400)
loader4 = DataLoader(dataset4, batch_size=batch_size, shuffle=True)

# loaders = zip(loader1, loader2, loader3, loader4)
#dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], acc=5)
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#brain_dataset = anatomy_data('data/brain/brain_singlecoil_train.mat', acc=5)
#brain_loader = DataLoader(brain_dataset, batch_size=batch_size, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.8)
save_dir = "universal_LDA/universal/checkpoints_sampling_4_8_12_15"

start_epoch = 1
start_phase = 3

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
        for i, (data1, data2, data3, data4) in enumerate(zip(loader1, loader2, loader3, loader4)):
            # undersampled image, k-space, mask, original image, original k-space
            #im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
            im_und1, k_und1, mask1, img_gnd1, k_gnd1 = data1
            # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
            
            im_und1 = im_und1.to(device)
            k_und1 = k_und1.to(device)
            mask1 = mask1.to(device)
            img_gnd1 = img_gnd1.to(device)
            k_gnd1 = k_gnd1.to(device)
            
            # forward pass
            optim.zero_grad()
            output1 = model(im_und1, k_und1, mask1, sampling_rates[0])
            output1 = torch.abs(output1[-1]).clamp(0, 1)
            img_gnd1 = torch.abs(img_gnd1)
            
            loss1 = torch.sum(torch.square(output1 - img_gnd1))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
            loss1.backward()
            optim.step()
            
            
            
            im_und2, k_und2, mask2, img_gnd2, k_gnd2 = data2
            im_und2 = im_und2.to(device)
            k_und2 = k_und2.to(device)
            mask2 = mask2.to(device)
            img_gnd2 = img_gnd2.to(device)
            k_gnd2 = k_gnd2.to(device)
            
            # forward pass
            optim.zero_grad()
            output2 = model(im_und2, k_und2, mask2, sampling_rates[1])
            output2 = torch.abs(output2[-1]).clamp(0, 1)
            img_gnd2 = torch.abs(img_gnd2)
            
            loss2 = torch.sum(torch.square(output2 - img_gnd2))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
            loss2.backward()
            optim.step()
            
            optim.zero_grad()
            im_und3, k_und3, mask3, img_gnd3, k_gnd3 = data3
            im_und3 = im_und3.to(device)
            k_und3 = k_und3.to(device)
            mask3 = mask3.to(device)
            img_gnd3 = img_gnd3.to(device)
            k_gnd3 = k_gnd3.to(device)
            
            # forward pass
            #optim.zero_grad()
            output3 = model(im_und3, k_und3, mask3, sampling_rates[2])
            output3 = torch.abs(output3[-1]).clamp(0, 1)
            img_gnd3 = torch.abs(img_gnd3)
            
            loss3 = torch.sum(torch.square(output3 - img_gnd3))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
            #loss = loss1 + loss2 + loss3
            loss3.backward()
            #loss2.backward()
            optim.step()

            
            optim.zero_grad()
            im_und4, k_und4, mask4, img_gnd4, k_gnd4 = data4
            im_und4 = im_und4.to(device)
            k_und4 = k_und4.to(device)
            mask4 = mask4.to(device)
            img_gnd4 = img_gnd4.to(device)
            k_gnd4 = k_gnd4.to(device)
            
            # forward pass
            #optim.zero_grad()
            output4 = model(im_und4, k_und4, mask4, sampling_rates[3])
            output4 = torch.abs(output4[-1]).clamp(0, 1)
            img_gnd4 = torch.abs(img_gnd4)
            
            loss4 = torch.sum(torch.square(output4 - img_gnd4))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
            loss4.backward()
            optim.step()
            loss = loss1 + loss2 + loss3 + loss4
            # loss = loss1 + loss2
            loss_list.append(loss.item())
            for j in range(batch_size):
                PSNR_list.append(psnr(np.abs(output1[j].squeeze().cpu().detach().numpy()), img_gnd1[j].squeeze().cpu().detach().numpy(), data_range=1))
                PSNR_list.append(psnr(np.abs(output2[j].squeeze().cpu().detach().numpy()), img_gnd2[j].squeeze().cpu().detach().numpy(), data_range=1))
                PSNR_list.append(psnr(np.abs(output3[j].squeeze().cpu().detach().numpy()), img_gnd3[j].squeeze().cpu().detach().numpy(), data_range=1))
                PSNR_list.append(psnr(np.abs(output4[j].squeeze().cpu().detach().numpy()), img_gnd4[j].squeeze().cpu().detach().numpy(), data_range=1))
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