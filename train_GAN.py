import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

from utils.dataset import universal_data
from utils.general import init_seeds
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np

from utils.GAN_model import *

init_seeds()
generator = PIGANGenerator()
discriminator = PIGANDiscriminator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Load the dataset
mask = "cartesian"
acc = 5
batch_size = 8
dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 
                          'data/knee/knee_singlecoil_train.mat'], 
                         # 'data/cardiac/cardiac_singlecoil_train.mat'], 
                         acc=acc, mask=mask)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.99)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.99)

save_dir = f"GAN/universal/checkpoints_{acc}_sampling_{mask}"

start_epoch = 1
n_epoch = 200

if os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    generator.load_state_dict(checkpoint['generator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G'])
    
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    scheduler_D.load_state_dict(checkpoint['scheduler_D'])
    
    start_epoch = checkpoint['epoch']
    print('Model loaded from checkpoint')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for epoch_i in range(start_epoch, n_epoch+1):
    loss_list = []
    PSNR_list = []
    for i, data in enumerate(loader):
        # undersampled image, k-space, mask, original image, original k-space
        im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
        # print(im_und.shape, k_und.shape, mask.shape, img_gnd.shape, k_gnd.shape)
        
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)
        
        
        fake_images = generator(im_und)
        
        # train discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1, 1, 1).to(device)  # Labels for real images
        fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)  # Labels for fake images
        
        # real image loss
        real_output = discriminator(img_gnd)#.squeeze()
        #d_loss_real = discriminator_loss(real_output, real_labels)
        
        # fake image loss
        fake_output = discriminator(fake_images.detach())#.squeeze()
        #d_loss_fake = discriminator_loss(fake_output, fake_labels)
        
        
        # total discriminator loss
        d_loss = discriminator_loss(real_output, fake_output, real_labels, fake_labels)
        d_loss.backward()
        optimizer_D.step()
        
        if (d_loss > 1e12).any():
            print("Fake output min/max:", fake_output.min(), fake_output.max())
            #print("Real labels min/max:", real_labels.min(), real_labels.max())
            print("Fake output contains NaN:", torch.isnan(fake_output).any())
            #print("Real labels contains NaN:", torch.isnan(real_labels).any())
        
        # train generator
        generator.zero_grad()
        fake_output = discriminator(fake_images).squeeze()
        
        adv_loss = adversarial_loss(fake_output)
        recon_loss = MAE_Loss(fake_images, img_gnd)
        fake_k_space = torch.fft.fft2(fake_images, norm='ortho')
        M_loss = MAE_Loss(fake_k_space * mask, k_gnd * mask)
        M_inv_loss = MAE_Loss(fake_k_space * (1-mask), k_gnd * (1-mask))
        
        generator_loss = adv_loss + recon_loss + 10 * M_loss + 10 * M_inv_loss
        generator_loss.backward()
        optimizer_G.step()
        
        loss_list.append(generator_loss.item())
    
        for j in range(batch_size):
            PSNR_list.append(psnr(np.abs(fake_images[j].squeeze().cpu().detach().numpy()), 
                                  np.abs(img_gnd[j].squeeze().cpu().detach().numpy()), data_range=1))
        if (i+1) % 100 == 0:
            print(i+1, generator_loss.item(), d_loss.item())
    avg_l = np.mean(loss_list)
    avg_p = np.mean(PSNR_list)
    epoch_data = '[Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
        (epoch_i, n_epoch, avg_l, avg_p)
    print(epoch_data)
    
    scheduler_G.step()
    scheduler_D.step()
    
    if epoch_i % 10 == 0:
        checkpoint = {
            'epoch': epoch_i,
            'generator': generator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'scheduler_D': scheduler_D.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))