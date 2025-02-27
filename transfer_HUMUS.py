from utils.humus_net import HUMUSNet
from utils.dataset import anatomy_data
from utils.general import init_seeds

import os
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader


init_seeds()
n_epoch = 50

batch_size  = 2
mask = 'cartesian'
acc = 4
anatomy = 'brain'
mode = 'sampling' # could be 'sampling' or 'anatomy

model = HUMUSNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


if mode == 'anatomy':
    model.load_state_dict(torch.load(
        f'HUMUS_Net/universal/checkpoints_{acc}_sampling_{mask}/checkpoint.pth')['state_dict'],
                        strict=False)
elif mode == 'sampling':
    model.load_state_dict(torch.load(
        f'HUMUS_Net/universal/checkpoints_{anatomy}_cross_sampling_{mask}/checkpoint.pth')['state_dict'])



anatomy = ['brain']

n_samples = 100

transfer_dataset = anatomy_data(f'data/{anatomy[0]}/{anatomy[0]}_singlecoil_train.mat', acc=acc, n=n_samples, mask=mask)
print(f"number of samples in {anatomy[0]} dataset: ", len(transfer_dataset))
transfer_loader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-5)
#scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
save_dir = f"HUMUS_Net/{anatomy[0]}/checkpoints_transfer_{acc}_sampling_{mask}_samples_{n_samples}"
if os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    #scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    
    print('Model loaded from checkpoint')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_epoch = 1

PSNR_list = []
loss_list = []

for epoch_i in range(start_epoch, n_epoch+1):
    for i, data in enumerate(transfer_loader):
        # undersampled image, k-space, mask, original image, original k-space
        im_und, k_und, mask, img_gnd, k_gnd = data
        # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
        
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)
        
        # forward pass
        optim.zero_grad()
        output = model(k_und, mask)
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
                    }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint.pth'))
# scheduler.step()
torch.save(checkpoint, os.path.join(save_dir, f'checkpoint.pth'))