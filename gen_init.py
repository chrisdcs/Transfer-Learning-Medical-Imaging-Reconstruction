import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.dataset import anatomy_data

from utils.model import Init_CNN
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import scipy.io as scio
import matplotlib.pyplot as plt


model = Init_CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('universal_LDA/brain/initNet_10/model_100.pt'))

model.to(device)
acc = 10
anatomy_train = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=acc, n=1000)
anatomy_test = anatomy_data(f'data/brain/brain_singlecoil_test.mat', acc=acc, n=1000)

train_loader = DataLoader(anatomy_train, batch_size=1, shuffle=False)
test_loader = DataLoader(anatomy_test, batch_size=1, shuffle=False)

init_data = {'k_space': None, 'images': None, 'inits': None, 'masks': None}
k_space = []
inits = []
images = []
masks = []
for i, data in enumerate(train_loader):
    _, k_und, mask, im_gnd, k_gnd = data
    
    k_und = k_und.to(device)
    k_gnd = k_gnd.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        output = model(k_und, mask)
    
    output = output.cpu().detach().numpy().squeeze()
    recon = np.abs(np.fft.ifft2(output, norm='ortho'))
    
    #k_space.append(k_und.cpu().detach().numpy().squeeze())
    k_space.append(output)
    images.append(im_gnd.numpy().squeeze())
    inits.append(recon)
    masks.append(mask.cpu().detach().numpy().squeeze())

init_data['k_space'] = np.array(k_space)
init_data['images'] = np.array(images)
init_data['masks'] = np.array(masks)
init_data['inits'] = np.array(inits)

scio.savemat('data/brain/brain_singlecoil_train_init_10.mat', init_data)


init_data_test = {'k_space': None, 'images': None, 'inits': None, 'masks': None}
k_space = []
images = []
inits = []
masks = []

for i, data in enumerate(test_loader):
    _, k_und, mask, img_gnd, k_gnd = data
    
    k_und = k_und.to(device)
    k_gnd = k_gnd.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        output = model(k_und, mask)
    
    output = output.cpu().detach().numpy().squeeze()
    recon = np.abs(np.fft.ifft2(output, norm='ortho'))
    #k_space.append(k_und.cpu().detach().numpy().squeeze())
    k_space.append(output)
    images.append(img_gnd.numpy().squeeze())
    inits.append(recon)
    masks.append(mask.cpu().detach().numpy().squeeze())
    
init_data_test['k_space'] = np.array(k_space)
init_data_test['images'] = np.array(images)
init_data_test['masks'] = np.array(masks)
init_data_test['inits'] = np.array(inits)

scio.savemat(f'data/brain/brain_singlecoil_test_init_{acc}.mat', init_data_test)
