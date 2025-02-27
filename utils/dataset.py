import torch.nn as nn
import scipy.io as scio
from torch.utils.data import Dataset
import numpy as np
import utils.compressed_sensing as cs


class anatomy_data(Dataset):
    def __init__(self, file, acc, n, mask=None):
        # acc: acceleration rate
        self.data = scio.loadmat(file)
        self.acc = acc
        self.n = n
        self.mask = mask

    def __len__(self):
        return min(self.n,len(self.data['images']))

    def __getitem__(self, idx):
        # return undersampled image, k-space, mask, original image, original k-space
        image = self.data['images'][idx][None,:,:]
        k_space = self.data['k_space'][idx][None,:,:]
        
        if not self.mask or self.mask == 'cartesian':
            mask = cs.cartesian_mask(image.shape, acc=self.acc, centred=True)
        elif self.mask == 'radial':
            mask = cs.radial_mask(image.shape, acc=self.acc, centred=True)
        im_und, k_und = cs.undersample(image, mask, centred=True)
        
        return im_und.astype(np.complex64), k_und.astype(np.complex64), mask.astype(np.float32), image.astype(np.complex64), k_space.astype(np.complex64)
    
class init_data(Dataset):
    def __init__(self, file, n):
        # acc: acceleration rate
        self.data = scio.loadmat(file)
        self.n = n
    
    def __len__(self):
        return min(self.n,len(self.data['images']))
    
    def __getitem__(self, idx):
        # return undersampled image, k-space, mask, original image, original k-space
        image = self.data['images'][idx][None,:,:]
        k_space = self.data['k_space'][idx][None,:,:]
        mask = self.data['masks'][idx][None,:,:]
        init = self.data['inits'][idx][None,:,:]
        
        #init_img = np.fft.ifft2(k_space, norm='ortho')
        #init_img = np.abs(init_img)
        #init_img = (init_img - np.min(init_img)) / (np.max(init_img) - np.min(init_img))
        
        return init.astype(np.float32), image.astype(np.complex64), k_space.astype(np.complex64), mask.astype(np.float32)
    
    
class universal_data(Dataset):
    def __init__(self, files, acc, mask=None):
        # acc: acceleration rate
        self.universal_image = []
        self.universal_k_space = []
        
        self.anatomy_names = []
        self.n_anatomy = len(files)
        
        self.mask = mask
        
        anatomies = []
        for file in files:
            anatomies.append(scio.loadmat(file))
            self.anatomy_names.append(file.split('/')[1])
        
        for j in range(self.n_anatomy):
            for i in range(min(400,len(anatomies[j]['images']))):
                self.universal_image.append(anatomies[j]['images'][i])
                self.universal_k_space.append(anatomies[j]['k_space'][i])
        
        self.acc = acc

    def __len__(self):
        return len(self.universal_image)

    def __getitem__(self, idx):
        # return undersampled image, k-space, mask, original image, original k-space
        image = self.universal_image[idx][None,:,:]
        k_space = self.universal_k_space[idx][None,:,:]
        
        if self.mask == 'radial':
            mask = cs.radial_mask(image.shape, acc=self.acc, centred=True)
        else:
            mask = cs.cartesian_mask(image.shape, acc=self.acc, centred=True)
        im_und, k_und = cs.undersample(image, mask, centred=True)
        
        return im_und.astype(np.complex64), k_und.astype(np.complex64), mask.astype(np.float32), image.astype(np.complex64), k_space.astype(np.complex64), self.anatomy_names[idx % self.n_anatomy]
    
    
class universal_sampling_data(Dataset):
    def __init__(self, file, sampling_rates, mask):
        self.universal_image = []
        self.universal_k_space = []
        
        self.sampling_rates = []
        self.n_sampling = len(sampling_rates)
        
        self.mask = mask
        
        sampling_data = []
        sampling_data.append(scio.loadmat(file))
        n = len(sampling_data[0]['images'])
        n = min(300, n)
        self.n = n
        for rate in sampling_rates:
            self.sampling_rates.append(rate)
        
        #for j in range(self.n_sampling):
        for i in range(self.n):
            image = sampling_data[0]['images'][i]
            k_space = sampling_data[0]['k_space'][i]
            
            self.universal_image.append(image)
            self.universal_k_space.append(k_space)
                
        
    def __len__(self):
        return len(self.universal_image) * self.n_sampling
    
    
    def __getitem__(self, idx):
        sampling_index = idx // self.n
        idx = idx % self.n
        
        image = self.universal_image[idx][None,:,:]
        k_space = self.universal_k_space[idx][None,:,:]
        
        if self.mask == 'radial':
            mask = cs.radial_mask(image.shape, acc=self.sampling_rates[sampling_index], centred=True)
        else:
            mask = cs.cartesian_mask(image.shape, acc=self.sampling_rates[sampling_index], centred=True)
        im_und, k_und = cs.undersample(image, mask, centred=True)
        
        return im_und.astype(np.complex64), k_und.astype(np.complex64), mask.astype(np.float32), image.astype(np.complex64), k_space.astype(np.complex64), self.sampling_rates[sampling_index]