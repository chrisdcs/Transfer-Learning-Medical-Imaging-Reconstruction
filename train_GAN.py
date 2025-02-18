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
optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.99)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.99)

save_dir = f"GAN/universal/checkpoints_{acc}_sampling_{mask}"

start_epoch = 1
n_epoch = 300

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

#real_labels = torch.ones(batch_size, 1, 1, 1).to(device) * 0.9  # Labels for real images
#fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device) + 0.1  # Labels for fake images

def add_gaussian_noise(x, noise_std=0.01):
    noise = torch.randn_like(x) * noise_std
    return x + noise

for epoch_i in range(start_epoch, n_epoch+1):
    loss_list = []
    PSNR_list = []
    for i, data in enumerate(loader):
        # undersampled image, k-space, mask, original image, original k-space
        im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)
        
        # train discriminator
        discriminator.zero_grad()
        real_output = discriminator(add_gaussian_noise(img_gnd))
        fake_images = generator(im_und, k_und, mask).detach()
        fake_output = discriminator(add_gaussian_noise(fake_images))
        d_loss = discriminator_loss(real_output, fake_output)
        d_loss.backward()
        optimizer_D.step()

        # train generator
        #for _ in range(2):
        generator.zero_grad()
        fake_images = generator(im_und, k_und, mask)
        fake_output = discriminator(add_gaussian_noise(fake_images))#.squeeze()
        adv_loss = adversarial_loss(fake_output)
        recon_loss = MAE_Loss(fake_images, img_gnd)
        fake_k_space = torch.fft.fft2(fake_images, norm='ortho')
        M_loss = MAE_Loss(fake_k_space * mask, k_gnd * mask)
        M_inv_loss = MAE_Loss(fake_k_space * (1-mask), k_gnd * (1-mask))
        generator_loss = adv_loss + 10 * recon_loss + 5 * M_loss + 5 * M_inv_loss
        generator_loss.backward()
        optimizer_G.step()

        loss_list.append(generator_loss.item())
        
        # PSNR
        for j in range(batch_size):
            PSNR_list.append(psnr(np.abs(fake_images[j].squeeze().cpu().detach().numpy()), 
                                  np.abs(img_gnd[j].squeeze().cpu().detach().numpy()), data_range=1))
        if (i+1) % 10 == 0:
            print(i+1, f"adver {adv_loss.item():.3f}, recon {recon_loss.item():.3f}, M {M_loss.item():.3f}, M_inv {M_inv_loss.item():.3f}, discriminator {d_loss.item():.3f}")
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