import torch
import torch.nn as nn
import os
from utils.model import data_consistency


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.dataset import *

from utils.general import init_seeds
from utils.UMRI_model import DnCn, UDnCn



#brain_model = DnCn()
#knee_model = DnCn()

model_10 = DnCn()
model_5 = DnCn()
model_3 = DnCn()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mask = "cartesian"
#acc = 5
#brain_model.load_state_dict(torch.load(os.path.join("universal_MRI", "brain", 
#                                                    f"{mask}_{acc}", 'model_100.pth')))
#knee_model.load_state_dict(torch.load(os.path.join("universal_MRI", "knee", 
#                                                   f"{mask}_{acc}", 'model_100.pth')))

#brain_model.to(device)
#knee_model.to(device)

model_10.load_state_dict(torch.load(
    os.path.join("universal_MRI", "brain", f"{mask}_10", 'model_100.pth')))
model_5.load_state_dict(torch.load(
    os.path.join("universal_MRI", "brain", f"{mask}_5", 'model_100.pth')))
model_3.load_state_dict(torch.load(
    os.path.join("universal_MRI", "brain", f"{mask}_3.33", 'model_100.pth')))
model_10.to(device)
model_5.to(device)
model_3.to(device)
model_10.eval()
model_5.eval()
model_3.eval()

#universal_model = UDnCn(['brain', 'knee'])

universal_model = UDnCn(['10', '5', '3'])

#universal_model.load_state_dict(torch.load(os.path.join("universal_MRI", "universal", f"{mask}_{acc}", 'model_200.pth')))
universal_model.load_state_dict(torch.load(os.path.join("universal_MRI", "universal", f"brain_{mask}", 'model_200.pth')))
universal_model.eval()
universal_model.to(device)


def attention_loss(feat1, feat2):
    O1 = torch.sum(torch.abs(feat1), dim=1, keepdim=True)
    O2 = torch.sum(torch.abs(feat2), dim=1, keepdim=True)
    
    # normalize O1 and O2 other than the batch dimension
    O1_norm = O1 / torch.norm(O1, dim = (2, 3), keepdim=True)
    O2_norm = O2 / torch.norm(O2, dim = (2, 3), keepdim=True)
    
    return torch.sum(torch.abs(O1_norm - O2_norm))


acc = 5
mask = 'cartesian'
#dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 
#                          'data/knee/knee_singlecoil_train.mat'], 
#                         acc=acc, mask=mask)
dataset = universal_sampling_data(
    'data/brain/brain_singlecoil_train.mat',[10, 5, 3.33], "cartesian")
                                   
loader = DataLoader(dataset, batch_size=1, shuffle=False)

optim = torch.optim.Adam(universal_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.95)

n_epochs = 200

#folder = os.path.join("universal_MRI", "universal", f"{mask}_{acc}_MD")
folder = os.path.join("universal_MRI", "universal", f"brain_{mask}_MD")
if not os.path.exists(folder):
    os.makedirs(folder)

for epoch in range(1, n_epochs+1):
    
    PSNR_list = []
    loss_list = []
    
    for i, data in enumerate(loader):
        
        im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
        
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)    
        
        if torch.is_tensor(anatomy[0]):
            if anatomy[0].item() == 3.33:
                anatomy = ['3']
            else:
                anatomy = [str(anatomy[0].item())]
        
        optim.zero_grad()
        output, feats = universal_model(im_und, k_und, mask, anatomy[0], MD=True)
        output = torch.abs(output).clamp(0, 1)
        
        #if anatomy[0] == 'brain':
        #    _, feats_ind = brain_model(im_und, k_und, mask, MD=True)
        #elif anatomy[0] == 'knee':
        #    _, feats_ind = knee_model(im_und, k_und, mask, MD=True)
        
        if anatomy[0] == '3':
            _, feats_ind = model_3(im_und, k_und, mask, MD=True)
        elif anatomy[0] == '5':
            _, feats_ind = model_5(im_und, k_und, mask, MD=True)
        elif anatomy[0] == '10':
            _, feats_ind = model_10(im_und, k_und, mask, MD=True)
        
        img_gnd = torch.abs(img_gnd)
        
        loss = torch.sum(torch.square(output - img_gnd))
        AL = 0
        for feat1, feat2 in zip(feats, feats_ind):
            AL += attention_loss(feat1, feat2)
        #print(loss.data, AL.data)
        #break
        loss += AL * 1e-3
        
        loss.backward()
        optim.step()
        
        loss_list.append(loss.item())
        
        for j in range(output.shape[0]):
            PSNR_list.append(psnr(output[j].cpu().detach().numpy(), img_gnd[j].cpu().detach().numpy()))
        if (i+1) % 100 == 0:
            print(i+1, loss.item())
    scheduler.step()
    avg_l = np.mean(loss_list)
    avg_p = np.mean(PSNR_list)
    epoch_data = ' [Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
            (epoch, n_epochs, avg_l, avg_p)
    print(epoch_data)
    
    if (epoch) % 10 == 0:
        torch.save(universal_model.state_dict(), os.path.join(folder, 'model_%d.pth' % (epoch)))
        print('Model saved\n')