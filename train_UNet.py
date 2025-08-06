from utils.dataset import universal_data, universal_sampling_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import numpy as np

from utils.UNet_model import *
import os

model = UNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
model.to(device)
mode = "domain" # could be changed to "anatomy" for cross-anatomy transfer learning
if mode == "anatomy":
    anatomies = ['brain', 'knee']
elif mode == "sampling":
    anatomies = ['10', '5', '3']
elif mode == "dataset":
    anatomies = ['imagenet']
elif mode == "domain":
    anatomies = ['imagenet', 'cifar10']

mask = 'cartesian'
acc = 5
if mode == "anatomy":
    dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], 
                            # 'data/cardiac/cardiac_singlecoil_train.mat'], 
                            acc=acc, mask=mask)
elif mode == "sampling":
    anatomy = 'brain'
    file = f'data/{anatomy}/brain_singlecoil_train.mat'
    # file = f'data/{anatomy}/knee_singlecoil_train.mat' # could be changed to other things like knee etc.
    dataset = universal_sampling_data(file, [10, 5, 3.33], "cartesian")
elif mode == "dataset":
    # this is for cross-dataset transfer learning
    anatomy = 'imagenet'
    file = f'data/{anatomy}/{anatomy}_singlecoil_train.mat' # could be changed to other things like knee etc.
    dataset = universal_data([file], acc=acc, mask = "cartesian", n=800)
elif mode == "domain":
    files = [f'data/{anatomy}/{anatomy}_singlecoil_train.mat' for anatomy in anatomies]
    dataset = universal_data(files, acc=acc, mask=mask, n=400)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#brain_dataset = anatomy_data('data/brain/brain_singlecoil_train.mat', acc=5)
#brain_loader = DataLoader(brain_dataset, batch_size=batch_size, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
if mode == "anatomy":
    save_dir = f"UNet/universal/cross_anatomy/checkpoints_{acc}_{mask}"
elif mode == "sampling":
    save_dir = f"UNet/universal/cross_sampling/checkpoints_{anatomy}_{mask}"
elif mode == "dataset":
    save_dir = f"UNet/universal/cross_dataset/checkpoints_{anatomy}_{mask}"
elif mode == "domain":
    save_dir = f"UNet/universal/cross_domain/checkpoints_{acc}_{mask}"
start_epoch = 1

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    
    scheduler.step()
    scheduler.step()
    scheduler.step()
    print('Model loaded from checkpoint')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

PSNR_list = []
loss_list = []

n_epoch = 200
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
        output = model(im_und, k_und, mask)
        output = torch.abs(output)
        img_gnd = torch.abs(img_gnd)
        
        loss = torch.sum(torch.square(output - img_gnd))#F.mse_loss(x_output.real, img_gnd.real) + F.mse_loss(x_output.imag, img_gnd.imag)
        loss.backward()
        optim.step()
        
        loss_list.append(loss.item())
    
        for j in range(batch_size):
            PSNR_list.append(psnr(np.abs(output[j].squeeze().cpu().detach().numpy()), img_gnd[j].squeeze().cpu().detach().numpy(), data_range=1))
        if (i+1) % 100 == 0:
            print(i+1, loss.item())
    avg_l = np.mean(loss_list)
    avg_p = np.mean(PSNR_list)
    epoch_data = '[Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
        (epoch_i, n_epoch, avg_l, avg_p)
    print(epoch_data)
    
    if epoch_i % 10 == 0:
        checkpoint = {
                            'epoch': epoch_i, 
                            'state_dict': model.state_dict(),
                            'optimizer': optim.state_dict(),
                            'scheduler': scheduler.state_dict()
                            }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint.pth'))
    scheduler.step()