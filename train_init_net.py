import torch
import numpy as np
from utils.dataset import anatomy_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch.nn.functional as F

import os
from utils.model import Init_CNN, complex_mse_loss


init_seeds()
anatomy = 'brain'
model = Init_CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 2
model.to(device)


anatomy_dataset1 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=4, n=200)
anatomy_dataset2 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=8, n=200)
anatomy_dataset3 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=12, n=200)
anatomy_dataset4 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=15, n=200)
anatomy_dataset5 = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=10, n=200)
anatomy_loader1 = DataLoader(anatomy_dataset1, batch_size=batch_size, shuffle=True)
anatomy_loader2 = DataLoader(anatomy_dataset2, batch_size=batch_size, shuffle=True)
anatomy_loader3 = DataLoader(anatomy_dataset3, batch_size=batch_size, shuffle=True)
anatomy_loader4 = DataLoader(anatomy_dataset4, batch_size=batch_size, shuffle=True)
anatomy_loader5 = DataLoader(anatomy_dataset5, batch_size=batch_size, shuffle=True)


optim = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.7)
save_dir = f"universal_LDA/{anatomy}/initNet"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
n_epoch = 100
for epoch_i in range(1, n_epoch+1):
    for i, (data1, data2, data3, data4, data5) in enumerate(zip(anatomy_loader1, anatomy_loader2, anatomy_loader3, anatomy_loader4, anatomy_loader5)):
        # undersampled image, k-space, mask, original image, original k-space
        #_, k_und, _, _, k_gnd = data
        _, k_und1, _, _, k_gnd1 = data1
        _, k_und2, _, _, k_gnd2 = data2
        _, k_und3, _, _, k_gnd3 = data3
        _, k_und4, _, _, k_gnd4 = data4
        _, k_und5, _, _, k_gnd5 = data5
        
        # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
        
        #im_und = im_und.to(device)
        #k_und = k_und.to(device)
        k_und1, k_und2, k_und3, k_und4, k_und5 = k_und1.to(device), k_und2.to(device), k_und3.to(device), k_und4.to(device), k_und5.to(device)
        k_gnd1, k_gnd2, k_gnd3, k_gnd4, k_gnd5 = k_gnd1.to(device), k_gnd2.to(device), k_gnd3.to(device), k_gnd4.to(device), k_gnd5.to(device)
        #mask = mask.to(device)
        #img_gnd = img_gnd.to(device)
        #k_gnd = k_gnd.to(device)    
        
        # forward pass
        optim.zero_grad()
        output1 = model(k_und1)
        output2 = model(k_und2)
        output3 = model(k_und3)
        output4 = model(k_und4)
        output5 = model(k_und5)
        
        #output = torch.abs(output[-1]).clamp(0, 1)
        
        loss = complex_mse_loss(output1, k_gnd1) + complex_mse_loss(output2, k_gnd2) + complex_mse_loss(output3, k_gnd3) + complex_mse_loss(output4, k_gnd4) + complex_mse_loss(output5, k_gnd5)
        loss.backward()
        optim.step()
        
        if i % 10 == 0:
            print(f"Epoch: {epoch_i}, Iter: {i}, Loss: {loss.item()}")
        
    scheduler.step()
    
    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), f"{save_dir}/model_{epoch_i}.pt")
        print(f"Model saved at epoch {epoch_i}")