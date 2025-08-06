import torch
import os
import torch.nn as nn

from utils.model import *

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.dataset import *

from utils.general import init_seeds




if os.path.exists('./data/brain/Meta_brain_singlecoil_train.mat'):
    print('Meta brain singlecoil dataset exists!')
else:
    brain_folder = './data/brain/'
    # load data
    train_data = scio.loadmat(os.path.join(brain_folder, 'brain_singlecoil_train.mat'))
    print(train_data['images'].shape)

    new_train_img = train_data['images'][:300,:,:]
    new_val_img = train_data['images'][300:400,:,:]

    new_train_kspace = train_data['k_space'][:300,:,:]
    new_val_kspace = train_data['k_space'][300:400,:,:]

    # create dataset
    new_data_brain = {'images': new_train_img, 'k_space': new_train_kspace}
    val_data_brain = {'images': new_val_img, 'k_space': new_val_kspace}

    scio.savemat('./data/brain/Meta_brain_singlecoil_train.mat', new_data_brain)
    scio.savemat('./data/brain/Meta_brain_singlecoil_val.mat', val_data_brain)


if os.path.exists('./data/knee/Meta_knee_singlecoil_train.mat'):
    print('Meta knee singlecoil dataset exists!')
else:
    # do the same for knee
    knee_folder = './data/knee/'
    # load data
    train_data = scio.loadmat(os.path.join(knee_folder, 'knee_singlecoil_train.mat'))
    print(train_data['images'].shape)

    new_train_img = train_data['images'][:300,:,:]
    new_val_img = train_data['images'][300:400,:,:]

    new_train_kspace = train_data['k_space'][:300,:,:]
    new_val_kspace = train_data['k_space'][300:400,:,:]

    # create dataset
    new_data_knee = {'images': new_train_img, 'k_space': new_train_kspace}
    val_data_knee = {'images': new_val_img, 'k_space': new_val_kspace}

    scio.savemat('./data/knee/Meta_knee_singlecoil_train.mat', new_data_knee)
    scio.savemat('./data/knee/Meta_knee_singlecoil_val.mat', val_data_knee)

init_seeds()

n_phase = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
mask = "cartesian"
acc = 5
mode = "domain"

if mode == "anatomy":
    anatomies = ['brain', 'knee']
    train_dataset = universal_data(['data/brain/Meta_brain_singlecoil_train.mat', 
                            'data/knee/Meta_knee_singlecoil_train.mat'], 
                            # 'data/cardiac/cardiac_singlecoil_train.mat'], 
                            acc=acc, mask=mask)
    val_dataset = universal_data(['data/brain/Meta_brain_singlecoil_val.mat',
                                'data/knee/Meta_knee_singlecoil_val.mat'], 
                                # 'data/cardiac/cardiac_singlecoil_val.mat'], 
                                acc=acc, mask=mask)
    save_dir = f"Meta/universal/cross_anatomy/checkpoints_{acc}_{mask}"
elif mode == "sampling":
    anatomies = ['10', '5', '3']
    anatomy = 'brain'
    train_file = f'data/{anatomy}/Meta_brain_singlecoil_train.mat'
    val_file = f'data/{anatomy}/Meta_brain_singlecoil_val.mat'
    train_dataset = universal_sampling_data(train_file, [10, 5, 3.33], mask)
    val_dataset = universal_sampling_data(val_file, [10, 5, 3.33], mask)
    save_dir = f"Meta/universal/cross_sampling/checkpoints_{anatomy}_{mask}"
elif mode == "dataset":
    anatomies = ['imagenet']
    train_dataset = universal_data(['data/imagenet/imagenet_singlecoil_train.mat'], acc=acc, mask = mask, n=800)
    val_dataset = universal_data(['data/imagenet/imagenet_singlecoil_train.mat'], acc=acc, mask=mask, n=100)
    save_dir = f"Meta/universal/cross_dataset/imagenet_{mask}_{acc}"
elif mode == "domain":
    anatomies = ['imagenet', 'cifar10']
    files = [f'data/{anatomy}/{anatomy}_singlecoil_train.mat' for anatomy in anatomies]
    train_dataset = universal_data(files, acc=acc, mask=mask, n=400)
    val_dataset = universal_data(files, acc=acc, mask=mask, n=100)
    save_dir = f"Meta/universal/cross_domain/checkpoints_{acc}_{mask}"
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_iter = iter(val_loader)


model = Meta(
    anatomies=anatomies,
    channel_num=16,
    n_block=15
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Separate parameters by name
h_dict_params = []
main_params = []
for name, param in model.named_parameters():
    if name.startswith('h_dict'):
        h_dict_params.append(param)
    else:
        main_params.append(param)
        

net_optim = torch.optim.Adam(main_params, lr=1e-4)
w_optim = torch.optim.Adam(h_dict_params, lr=1e-4)

scheduler_net = torch.optim.lr_scheduler.StepLR(net_optim, step_size=10, gamma=0.5)
scheduler_w = torch.optim.lr_scheduler.StepLR(w_optim, step_size=10, gamma=0.7)


start_epoch = 1
start_phase = 3

n_epoch = 20

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    net_optim.load_state_dict(checkpoint['net_optim'])
    w_optim.load_state_dict(checkpoint['w_optim'])
    scheduler_net.load_state_dict(checkpoint['scheduler_net'])
    scheduler_w.load_state_dict(checkpoint['scheduler_w'])
    start_epoch = checkpoint['epoch']
    start_phase = checkpoint['phase']
    
    #scheduler.step()
    #scheduler.step()
    #scheduler.step()
    for i in range(30):
        scheduler_net.step()
        scheduler_w.step()
    print('Model loaded from checkpoint')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
for PhaseNo in range(start_phase, n_phase+1, 2):
    model.set_PhaseNo(PhaseNo)
    PSNR_list = []
    loss_list = []
    
    for epoch_i in range(start_epoch, n_epoch+1):
        for i, data in enumerate(train_loader):
            # undersampled image, k-space, mask, original image, original k-space
            im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
            # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
            
            im_und = im_und.to(device)
            k_und = k_und.to(device)
            mask = mask.to(device)
            img_gnd = img_gnd.to(device)
            k_gnd = k_gnd.to(device)
            
            # forward pass
            for _ in range(3):
                net_optim.zero_grad()
                
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
                net_optim.step()
                
                loss_list.append(loss.item())

            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            
            im_und, k_und, mask, img_gnd, k_gnd, anatomy = val_batch
            
            im_und = im_und.to(device)
            k_und = k_und.to(device)
            mask = mask.to(device)
            img_gnd = img_gnd.to(device)
            k_gnd = k_gnd.to(device)
            
            w_optim.zero_grad()
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
            w_optim.step()
            
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
                                'net_optim': net_optim.state_dict(),
                                'w_optim': w_optim.state_dict(),
                                'scheduler_net': scheduler_net.state_dict(),
                                'scheduler_w': scheduler_w.state_dict()
                             }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint.pth'))
    start_epoch = 1
    scheduler_net.step()
    scheduler_w.step()
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{PhaseNo}.pth'))
    