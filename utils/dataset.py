import torch.nn as nn
import scipy.io as scio
from torch.utils.data import Dataset
import numpy as np
import utils.compressed_sensing as cs


class anatomy_data(Dataset):
    def __init__(self, file, acc, n):
        # acc: acceleration rate
        self.data = scio.loadmat(file)
        self.acc = acc
        self.n = n

    def __len__(self):
        return min(self.n,len(self.data['images']))

    def __getitem__(self, idx):
        # return undersampled image, k-space, mask, original image, original k-space
        image = self.data['images'][idx][None,:,:]
        k_space = self.data['k_space'][idx][None,:,:]
        
        mask = cs.cartesian_mask(image.shape, acc=self.acc, centred=True)
        im_und, k_und = cs.undersample(image, mask, centred=True)
        
        return im_und.astype(np.complex64), k_und.astype(np.complex64), mask.astype(np.float32), image.astype(np.complex64), k_space.astype(np.complex64)
    
    
    
class universal_data(Dataset):
    def __init__(self, files, acc):
        # acc: acceleration rate
        self.universal_image = []
        self.universal_k_space = []
        
        self.anatomy_names = []
        self.n_anatomy = len(files)
        
        anatomies = []
        for file in files:
            anatomies.append(scio.loadmat(file))
            self.anatomy_names.append(file.split('/')[1])
        
        for i in range(400):
            for j in range(self.n_anatomy):
                self.universal_image.append(anatomies[j]['images'][i])
                self.universal_k_space.append(anatomies[j]['k_space'][i])
        
        self.acc = acc

    def __len__(self):
        return 800#len(self.data['images'])

    def __getitem__(self, idx):
        # return undersampled image, k-space, mask, original image, original k-space
        image = self.universal_image[idx][None,:,:]
        k_space = self.universal_k_space[idx][None,:,:]
        
        mask = cs.cartesian_mask(image.shape, acc=self.acc, centred=True)
        im_und, k_und = cs.undersample(image, mask, centred=True)
        
        return im_und.astype(np.complex64), k_und.astype(np.complex64), mask.astype(np.float32), image.astype(np.complex64), k_space.astype(np.complex64), self.anatomy_names[idx % self.n_anatomy]