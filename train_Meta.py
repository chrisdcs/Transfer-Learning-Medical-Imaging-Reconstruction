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


class Meta(nn.Module):
    
    def __init__(self, **kwargs):
        super(Meta, self).__init__()
        anatomies = kwargs['anatomies']
        channel_num = kwargs['channel_num']
        self.channel_num = channel_num
        # channel_num = 32
        self.h_dict = nn.ParameterDict(
            {
                anatomy: 
                nn.Parameter(torch.Tensor([1.0, 1.0]), requires_grad=True) 
                for anatomy in anatomies
            }
        )
        
        cur_iter = kwargs['n_block']
        self.n_block = cur_iter
        self.cur_iter = cur_iter
        
        # self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        self.soft_thr = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.Tensor([0.002]), requires_grad=True) for anatomy in anatomies
                        })
        
        # self.alphas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        # self.betas = nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True)
        
        self.alphas = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True) for anatomy in anatomies
                        })
        self.betas = nn.ParameterDict({
                            anatomy: nn.Parameter(torch.tensor([1e-12] * kwargs['n_block']), requires_grad=True) for anatomy in anatomies
                        })
        
        # complex learnable blocks are still the same
        # except the gradient part
        # customize gradient function under universal LDA
        self.ImgNet = Complex_Learnable_Block(
            n_feats=channel_num,
            n_convs=4,
            k_size=3,
            padding=1,
        )
        
    def set_PhaseNo(self, cur_iter):
        self.cur_iter = cur_iter
        
    def add_anatomy(self, name, out_feats=16):
        self.h_dict[name] = nn.Parameter(torch.Tensor([1.0, 1.0]), requires_grad=True)
        #.to(next(self.parameters()).device)
        
        self.soft_thr[name] = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)#.to(next(self.parameters()).device)
        self.alphas[name] = nn.Parameter(torch.tensor([1e-12] * self.n_block), requires_grad=True)#.to(next(self.parameters()).device)
        self.betas[name] = nn.Parameter(torch.tensor([1e-12] * self.n_block), requires_grad=True)#.to(next(self.parameters()).device)
    
    def phase(self, x, k, phase, gamma, mask, anatomy, return_g=False):
        '''
            computation for each phase
        '''
        alpha = torch.abs(self.alphas[anatomy][phase])
        beta = torch.abs(self.betas[anatomy][phase])
        
        # update x
        #Ax = projection.apply(x, self.options)
        Fx = torch.fft.fft2(x, norm="ortho")
        # Fx = data_consistency(Fx, k, mask)
        residual = Fx - k
        # residual_S_new = Ax - z
        # grad_fx = projection_t.apply(residual_S_new, self.options)
        grad = torch.fft.ifft2(residual, norm="ortho")

        #c = x - alpha * grad_fx
        c = x - alpha * grad
        cache_x = self.ImgNet(c)
        
        grad_R = self.ImgNet.gradient(cache_x, gamma)
        sig = F.sigmoid(self.h_dict[anatomy])
        sig_times_grad = torch.complex(sig[0] * grad_R.real - sig[1] * grad_R.imag, 
                                       sig[1] * grad_R.real + sig[0] * grad_R.imag)
        
        u = c - beta * sig_times_grad
        
        Fu = torch.fft.fft2(u, norm="ortho")
        Fu = data_consistency(Fu, k, mask)
        u = torch.fft.ifft2(Fu, norm="ortho")
        
        if return_g:
            return u, cache_x[-1]
        return u
    
    def forward(self, x, k, mask, anatomy, return_g=None):
        x_list = []
        g_list = []
        for phase in range(self.cur_iter):
            if return_g:
                x, hg = self.phase(x, k, phase, 0.9**phase, mask, anatomy, return_g)
                g_list.append(hg)
            else:
                x = self.phase(x, k, phase, 0.9**phase, mask, anatomy)
            x_list.append(x)
        if return_g:
            return x_list, g_list
        return x_list

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
anatomies = ['brain', 'knee']#, 'cardiac']
n_phase = 15
model = Meta(
    anatomies=anatomies,
    channel_num=16,
    n_block=n_phase
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
model.to(device)


mask = "cartesian"

acc = 5

train_dataset = universal_data(['data/brain/Meta_brain_singlecoil_train.mat', 
                          'data/knee/Meta_knee_singlecoil_train.mat'], 
                         # 'data/cardiac/cardiac_singlecoil_train.mat'], 
                         acc=acc, mask=mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = universal_data(['data/brain/Meta_brain_singlecoil_val.mat',
                            'data/knee/Meta_knee_singlecoil_val.mat'], 
                             # 'data/cardiac/cardiac_singlecoil_val.mat'], 
                             acc=acc, mask=mask)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_iter = iter(val_loader)
# optimizer
net_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

anatomies = ['brain', 'knee']

model = Meta(
    anatomies=anatomies,
    channel_num=16,
    n_block=15
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Separate parameters by name
mask = "radial"
acc = 5
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
save_dir = f"Meta/universal/{mask}_{acc}"

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
    