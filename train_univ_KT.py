import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import utils.compressed_sensing as cs
import torch.nn as nn
import torch

import os
import torch.nn.functional as F

from utils.dataset import universal_data
from utils.general import init_seeds, normalized_cross_correlation
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.model import Universal_LDA_vis, LDA_vis

n_phase = 15
n_epoch = 50

init_seeds()
anatomies = ['brain', 'knee']

model = Universal_LDA_vis(n_block=n_phase, anatomies=anatomies)
anatomy_model_1 = LDA_vis(n_block=n_phase)
anatomy_model_2 = LDA_vis(n_block=n_phase)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
model.to(device)
anatomy_model_1.to(device)
anatomy_model_2.to(device)

anatomy_model_1.load_state_dict(torch.load('universal_LDA/brain/checkpoints/checkpoint.pth'))
anatomy_model_2.load_state_dict(torch.load('universal_LDA/knee/checkpoints/checkpoint.pth'))

anatomy_model_1.eval()
anatomy_model_2.eval()

dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], acc=5)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#brain_dataset = anatomy_data('data/brain/brain_singlecoil_train.mat', acc=5)
#brain_loader = DataLoader(brain_dataset, batch_size=batch_size, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
save_dir = "universal_LDA/universal_KT/checkpoints"

start_epoch = 1
start_phase = 3

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    start_phase = checkpoint['phase']
    
    #scheduler.step()
    #scheduler.step()
    #scheduler.step()
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
            output, g, _ = model(im_und, k_und, mask, anatomy[0])
            output = torch.abs(output[-1]).clamp(0, 1)
            img_gnd = torch.abs(img_gnd)
            if PhaseNo <= 7:
                with torch.no_grad():
                    if anatomy[0] == 'brain':
                        _, g_i = anatomy_model_1(im_und, k_und, mask)
                    else:
                        _, g_i = anatomy_model_2(im_und, k_und, mask)
                KT_loss = 1-normalized_cross_correlation(torch.abs(g[-1][-1]), torch.abs(g_i[PhaseNo-1][-1]),False)
            else:
                KT_loss = torch.tensor(0)
            loss = torch.sum(torch.square(output - img_gnd)) + KT_loss# * 0.7 ** (PhaseNo-3)
            loss.backward()
            optim.step()
            
            loss_list.append(loss.item())
        
            for j in range(batch_size):
                PSNR_list.append(psnr(np.abs(output[j].squeeze().cpu().detach().numpy()), img_gnd[j].squeeze().cpu().detach().numpy(), data_range=1))
            if (i+1) % 100 == 0:
                print(i+1, loss.item(), KT_loss.item())
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