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
model.load_state_dict(torch.load('universal_LDA/brain/initNet/model_100.pt'))

model.to(device)

anatomy_data1 = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=4, n=1000)
anatomy_data2 = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=8, n=1000)
anatomy_data3 = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=12, n=1000)
anatomy_data4 = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=15, n=1000)
anatomy_data5 = anatomy_data(f'data/brain/brain_singlecoil_train.mat', acc=10, n=1000)

anatomy_test1 = anatomy_data(f'data/brain/brain_singlecoil_test.mat', acc=4, n=1000)
anatomy_test2 = anatomy_data(f'data/brain/brain_singlecoil_test.mat', acc=8, n=1000)
anatomy_test3 = anatomy_data(f'data/brain/brain_singlecoil_test.mat', acc=12, n=1000)
anatomy_test4 = anatomy_data(f'data/brain/brain_singlecoil_test.mat', acc=15, n=1000)
anatomy_test5 = anatomy_data(f'data/brain/brain_singlecoil_test.mat', acc=10, n=1000)


train_loader1 = DataLoader(anatomy_data1, batch_size=1, shuffle=False)
train_loader2 = DataLoader(anatomy_data2, batch_size=1, shuffle=False)
train_loader3 = DataLoader(anatomy_data3, batch_size=1, shuffle=False)
train_loader4 = DataLoader(anatomy_data4, batch_size=1, shuffle=False)
train_loader5 = DataLoader(anatomy_data5, batch_size=1, shuffle=False)

test_loader1 = DataLoader(anatomy_test1, batch_size=1, shuffle=False)
test_loader2 = DataLoader(anatomy_test2, batch_size=1, shuffle=False)
test_loader3 = DataLoader(anatomy_test3, batch_size=1, shuffle=False)
test_loader4 = DataLoader(anatomy_test4, batch_size=1, shuffle=False)
test_loader5 = DataLoader(anatomy_test5, batch_size=1, shuffle=False)

init_data4 = {'k_space': None, 'images': None}
init_data8 = {'k_space': None, 'images': None}
init_data12 = {'k_space': None, 'images': None}
init_data15 = {'k_space': None, 'images': None}
init_data10 = {'k_space': None, 'images': None}
k_space4 = []
images4 = []
k_space8 = []
images8 = []
k_space12 = []
images12 = []
k_space15 = []
images15 = []
k_space10 = []
images10 = []

for i, (data1, data2, data3, data4, data5) in enumerate(zip(train_loader1, train_loader2, train_loader3, train_loader4, train_loader5)):
    _, k_und1, _, _, k_gnd1 = data1
    _, k_und2, _, _, k_gnd2 = data2
    _, k_und3, _, _, k_gnd3 = data3
    _, k_und4, _, _, k_gnd4 = data4
    _, k_und5, _, _, k_gnd5 = data5
    
    k_und1 = k_und1.to(device)
    k_und2 = k_und2.to(device)
    k_und3 = k_und3.to(device)
    k_und4 = k_und4.to(device)
    k_und5 = k_und5.to(device)
    
    k_gnd1 = k_gnd1.to(device)
    k_gnd2 = k_gnd2.to(device)
    k_gnd3 = k_gnd3.to(device)
    k_gnd4 = k_gnd4.to(device)
    k_gnd5 = k_gnd5.to(device)
    
    with torch.no_grad():
        output1 = model(k_und1)
        output2 = model(k_und2)
        output3 = model(k_und3)
        output4 = model(k_und4)
        output5 = model(k_und5)
    
    output1 = output1.cpu().detach().numpy().squeeze()
    output2 = output2.cpu().detach().numpy().squeeze()
    output3 = output3.cpu().detach().numpy().squeeze()
    output4 = output4.cpu().detach().numpy().squeeze()
    output5 = output5.cpu().detach().numpy().squeeze()
    
    recon1 = np.abs(np.fft.ifft2(output1))
    recon1 = (recon1 - np.min(recon1))/(np.max(recon1) - np.min(recon1))
    recon2 = np.abs(np.fft.ifft2(output2))
    recon2 = (recon2 - np.min(recon2))/(np.max(recon2) - np.min(recon2))
    recon3 = np.abs(np.fft.ifft2(output3))
    recon3 = (recon3 - np.min(recon3))/(np.max(recon3) - np.min(recon3))
    recon4 = np.abs(np.fft.ifft2(output4))
    recon4 = (recon4 - np.min(recon4))/(np.max(recon4) - np.min(recon4))
    recon5 = np.abs(np.fft.ifft2(output5))
    recon5 = (recon5 - np.min(recon5))/(np.max(recon5) - np.min(recon5))
    
    k_space4.append(output1)
    images4.append(recon1)
    
    k_space8.append(output2)
    images8.append(recon2)
    
    k_space12.append(output3)
    images12.append(recon3)
    
    k_space15.append(output4)
    images15.append(recon4)
    
    k_space10.append(output5)
    images10.append(recon5)

init_data4['k_space'] = np.array(k_space4)
init_data4['images'] = np.array(images4)
init_data8['k_space'] = np.array(k_space8)
init_data8['images'] = np.array(images8)
init_data12['k_space'] = np.array(k_space12)
init_data12['images'] = np.array(images12)
init_data15['k_space'] = np.array(k_space15)
init_data15['images'] = np.array(images15)
init_data10['k_space'] = np.array(k_space10)
init_data10['images'] = np.array(images10)

scio.savemat('data/brain/brain_singlecoil_train_init4.mat', init_data4)
scio.savemat('data/brain/brain_singlecoil_train_init8.mat', init_data8)
scio.savemat('data/brain/brain_singlecoil_train_init12.mat', init_data12)
scio.savemat('data/brain/brain_singlecoil_train_init15.mat', init_data15)
scio.savemat('data/brain/brain_singlecoil_train_init10.mat', init_data10)


init_data_test4 = {'k_space': None, 'images': None}
init_data_test8 = {'k_space': None, 'images': None}
init_data_test12 = {'k_space': None, 'images': None}
init_data_test15 = {'k_space': None, 'images': None}
init_data_test10 = {'k_space': None, 'images': None}
k_space4 = []
images4 = []
k_space8 = []
images8 = []
k_space12 = []
images12 = []
k_space15 = []
images15 = []
k_space10 = []
images10 = []


for i, (data1, data2, data3, data4, data5) in enumerate(zip(test_loader1, test_loader2, test_loader3, test_loader4, test_loader5)):
    _, k_und1, _, _, k_gnd1 = data1
    _, k_und2, _, _, k_gnd2 = data2
    _, k_und3, _, _, k_gnd3 = data3
    _, k_und4, _, _, k_gnd4 = data4
    _, k_und5, _, _, k_gnd5 = data5
    
    k_und1 = k_und1.to(device)
    k_und2 = k_und2.to(device)
    k_und3 = k_und3.to(device)
    k_und4 = k_und4.to(device)
    k_und5 = k_und5.to(device)
    
    k_gnd1 = k_gnd1.to(device)
    k_gnd2 = k_gnd2.to(device)
    k_gnd3 = k_gnd3.to(device)
    k_gnd4 = k_gnd4.to(device)
    k_gnd5 = k_gnd5.to(device)
    
    with torch.no_grad():
        output1 = model(k_und1)
        output2 = model(k_und2)
        output3 = model(k_und3)
        output4 = model(k_und4)
        output5 = model(k_und5)
    
    output1 = output1.cpu().detach().numpy().squeeze()
    output2 = output2.cpu().detach().numpy().squeeze()
    output3 = output3.cpu().detach().numpy().squeeze()
    output4 = output4.cpu().detach().numpy().squeeze()
    output5 = output5.cpu().detach().numpy().squeeze()
    
    recon1 = np.abs(np.fft.ifft2(output1))
    recon1 = (recon1 - np.min(recon1))/(np.max(recon1) - np.min(recon1))
    recon2 = np.abs(np.fft.ifft2(output2))
    recon2 = (recon2 - np.min(recon2))/(np.max(recon2) - np.min(recon2))
    recon3 = np.abs(np.fft.ifft2(output3))
    recon3 = (recon3 - np.min(recon3))/(np.max(recon3) - np.min(recon3))
    recon4 = np.abs(np.fft.ifft2(output4))
    recon4 = (recon4 - np.min(recon4))/(np.max(recon4) - np.min(recon4))
    
    recon5 = np.abs(np.fft.ifft2(output5))
    recon5 = (recon5 - np.min(recon5))/(np.max(recon5) - np.min(recon5))
    
    k_space4.append(output1)
    images4.append(recon1)
    
    k_space8.append(output2)
    images8.append(recon2)
    
    k_space12.append(output3)
    images12.append(recon3)
    
    k_space15.append(output4)
    images15.append(recon4)
    
    k_space10.append(output5)
    images10.append(recon5)
    
init_data_test4['k_space'] = np.array(k_space4)
init_data_test4['images'] = np.array(images4)
init_data_test8['k_space'] = np.array(k_space8)
init_data_test8['images'] = np.array(images8)
init_data_test12['k_space'] = np.array(k_space12)
init_data_test12['images'] = np.array(images12)
init_data_test15['k_space'] = np.array(k_space15)
init_data_test15['images'] = np.array(images15)
init_data_test10['k_space'] = np.array(k_space10)
init_data_test10['images'] = np.array(images10)

scio.savemat('data/brain/brain_singlecoil_test_init4.mat', init_data_test4)
scio.savemat('data/brain/brain_singlecoil_test_init8.mat', init_data_test8)
scio.savemat('data/brain/brain_singlecoil_test_init12.mat', init_data_test12)
scio.savemat('data/brain/brain_singlecoil_test_init15.mat', init_data_test15)
scio.savemat('data/brain/brain_singlecoil_test_init10.mat', init_data_test10)
