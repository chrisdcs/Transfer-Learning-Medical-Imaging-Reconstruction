import os
import nibabel as nib

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.dataset import *
from utils.model import *

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
anatomies = ['brain', 'knee']
n_phase = 15
model = Universal_LDA(n_block=n_phase, anatomies=anatomies, channel_num=16)
model.to(device)
model.load_state_dict(torch.load('universal_LDA/universal/checkpoints_10_sampling_radial/checkpoint.pth')['state_dict'])
brain_dataset = anatomy_data('data/brain/brain_singlecoil_test.mat', acc=10, n=200, mask='radial')
brain_loader = DataLoader(brain_dataset, batch_size=1, shuffle=True)

n_phase = 15
brain_model = LDA(n_block=n_phase, channel_num=16)
brain_model.to(device)
brain_model.load_state_dict(torch.load('universal_LDA/brain/checkpoints_10_sampling_radial/checkpoint.pth'))

n_phase = 15
knee_model = LDA(n_block=n_phase, channel_num=16)
knee_model.to(device)
knee_model.load_state_dict(torch.load('universal_LDA/knee/checkpoints_10_sampling_radial/checkpoint.pth'))

knee_dataset = anatomy_data('data/knee/knee_singlecoil_test.mat', acc=10, n=200, mask='radial')
knee_loader = DataLoader(knee_dataset, batch_size=1, shuffle=True)

def attention_loss(x1, x2):
    res = 0
    for i in range(len(x1)):
        O1 = F.normalize(x1[i], p=2, dim=1)
        O2 = F.normalize(x2[i], p=2, dim=1)
        res += torch.mean(torch.abs(O1 - O2))
    
    return res

def distillation_loss(x1, x2):
    res = 0
    for i in range(len(x1)):
        res += torch.mean(torch.abs(x1[i] - x2[i]))
        
    return res

knee_trainset = anatomy_data('data/knee/knee_singlecoil_train.mat', acc=10, n=400, mask='radial')
knee_train_loader = DataLoader(knee_trainset, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(50):
    print("epoch: ", epoch)
    for i, data in enumerate(knee_train_loader):
        # undersampled image, k-space, mask, original image, original k-space
        im_und, k_und, mask, img_gnd, k_gnd = data
        # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)    
        # forward pass
        with torch.no_grad():
            output, g2_list = knee_model(im_und, k_und, mask, True)
        output = torch.abs(output[-1]).clamp(0, 1)
        img_gnd = torch.abs(img_gnd)
        
        optimizer.zero_grad()
        univ_out, hg2_list = model(im_und, k_und, mask, 'knee', True)
        univ_out = torch.abs(univ_out[-1]).clamp(0, 1)
        
        # print(g2.shape, hg2.shape)
        loss = torch.sum(torch.square(univ_out - img_gnd)) + 0.1 * distillation_loss(g2_list, hg2_list)
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            print(loss.item())
            
PSNR_list = []
SSIM_list = []
with torch.no_grad():
    anatomy=('knee',)
    for i, data in enumerate(knee_loader):
        # undersampled image, k-space, mask, original image, original k-space
        im_und, k_und, mask, img_gnd, k_gnd = data
        # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)    
        # forward pass
        output = model(im_und, k_und, mask, anatomy[0])
        output = torch.abs(output[-1]).clamp(0, 1)
        img_gnd = torch.abs(img_gnd)
        

        p = psnr(np.abs(output[0].squeeze().cpu().detach().numpy()), img_gnd[0].squeeze().cpu().detach().numpy(), data_range=1)
        s = ssim(np.abs(output[0].squeeze().cpu().detach().numpy()), img_gnd[0].squeeze().cpu().detach().numpy(), data_range=1)
        PSNR_list.append(p)
        SSIM_list.append(s)
        print("psnr: ", p, "ssim: ", s)
        
        
print("universal knee (MD) average psnr: ", np.mean(PSNR_list))
print("universal knee (MD) average ssim: ", np.mean(SSIM_list))