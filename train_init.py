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

acc = 10
anatomy_dataset = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=acc, n=400)

anatomy_loader = DataLoader(anatomy_dataset, batch_size=batch_size, shuffle=True)


optim = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.7)
save_dir = f"universal_LDA/{anatomy}/initNet_{acc}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
n_epoch = 100
for epoch_i in range(1, n_epoch+1):
    for i, data in enumerate(anatomy_loader):
        _, k_und, mask, _, k_gnd = data
        
        #im_und = im_und.to(device)
        #k_und = k_und.to(device)
        k_und = k_und.to(device)
        k_gnd = k_gnd.to(device)
        mask = mask.to(device)
        #mask = mask.to(device)
        #img_gnd = img_gnd.to(device)
        #k_gnd = k_gnd.to(device)    
        
        # forward pass
        optim.zero_grad()
        output = model(k_und, mask)
        #output1 = model(k_und1, mask1)
        #output2 = model(k_und2, mask2)
        #output3 = model(k_und3, mask3)
        #output4 = model(k_und4, mask4)
        #output5 = model(k_und5, mask5)
        
        #output = torch.abs(output[-1]).clamp(0, 1)
        
        loss = complex_mse_loss(output, k_gnd)# + complex_mse_loss(output2, k_gnd2) + complex_mse_loss(output3, k_gnd3) + complex_mse_loss(output4, k_gnd4) + complex_mse_loss(output5, k_gnd5)
        loss.backward()
        optim.step()
        
        if i % 10 == 0:
            print(f"Epoch: {epoch_i}, Iter: {i}, Loss: {loss.item()}")
        
    scheduler.step()
    
    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), f"{save_dir}/model_{epoch_i}.pt")
        print(f"Model saved at epoch {epoch_i}")