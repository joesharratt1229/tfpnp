import os
import numpy as np
from PIL import Image

import torch 
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat

from tfpnp.data.util import scale_height, scale_width, data_augment
from tfpnp.utils import transforms
from tfpnp.utils.transforms import complex2real


class CSMRIDataset(Dataset):
    def __init__(self, datadir, fns, masks, noise_model=None, size=None, target_size=None, repeat=1, augment=False):
        super().__init__()
        self.datadir = datadir
        self.fns = fns or [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.fns = sorted(self.fns)        
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.augment = augment

    def __getitem__(self, index):
        #random selects one mask from sampling masks -> sampling of k space sdata
        mask = self.masks[np.random.randint(0, len(self.masks))]
        mask = mask.astype(np.bool)
        
        sigma_n = 0

        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size            
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)        

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2,0,1))
        else:
            raise NotImplementedError
        
        if self.augment:
            target = data_augment(target)

        #convert target and mask into torch tensors
        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)
        
        #transform into k space data 
        y0 = transforms.fft2_new(torch.stack([target, torch.zeros_like(target)], dim=-1))

        if self.noise_model is not None:
            #adds GWN noise to image
            y0, sigma_n = self.noise_model(y0)

        #zeros out portions of k-space data where mask equal to false
        y0[:, ~mask, :] = 0
        
        #degraded output transformed into spatial domain
        ATy0 = transforms.ifft2_new(y0)
        x0 = ATy0.clone().detach()

        #CONVERT into real valued tensor -> important when imaginary is close to 0
        output = complex2real(ATy0.clone().detach())
        mask = mask.unsqueeze(0).bool()
        #converts sigma into noise map in same dimensions as y0
        sigma_n = np.ones_like(y0) * sigma_n
        
        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'output': output, 'input': x0}
        
        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size


class CSMRIEvalDataset(Dataset):
    def __init__(self, datadir, masks, noise_model=None, size=None, target_size=None, repeat=1, augment=False):
        super().__init__()
        self.datadir = datadir
        self.fns = [im for im in os.listdir(self.datadir) if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png") or im.endswith(".tif")]      
        self.fns = sorted(self.fns)        
        self.masks = masks
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.augment = augment

    def __getitem__(self, index):
        #random selects one mask from sampling masks -> sampling of k space sdata
        mask = self.masks[np.random.randint(0, len(self.masks))]
        mask = mask.astype(np.bool)
        
        sigma_n = 0

        index = index % len(self.fns)
        imgpath = os.path.join(self.datadir, self.fns[index])
        target = Image.open(imgpath).convert('L')

        if self.target_size is not None:
            ow, oh = target.size            
            target = scale_height(target, self.target_size) if ow >= oh else scale_width(target, self.target_size)        

        target = np.array(target, dtype=np.float32) / 255.0

        if target.ndim == 2:
            target = target[None]
        elif target.ndim == 3:
            target = target.transpose((2,0,1))
        else:
            raise NotImplementedError
        
        if self.augment:
            target = data_augment(target)

        #convert target and mask into torch tensors
        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)
        
        #transform into k space data 
        y0 = transforms.fft2_new(torch.stack([target, torch.zeros_like(target)], dim=-1))

        if self.noise_model is not None:
            #adds GWN noise to image
            y0, sigma_n = self.noise_model(y0)

        #zeros out portions of k-space data where mask equal to false
        y0[:, ~mask, :] = 0
        
        #degraded output transformed into spatial domain
        ATy0 = transforms.ifft2_new(y0)
        x0 = ATy0.clone().detach()

        #CONVERT into real valued tensor -> important when imaginary is close to 0
        output = complex2real(ATy0.clone().detach())
        mask = mask.unsqueeze(0).bool()
        #converts sigma into noise map in same dimensions as y0
        sigma_n = np.ones_like(y0) * sigma_n
        
        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'output': output, 'input': x0}
        
        return dic

    def __len__(self):
        if self.size is None:
            return len(self.fns) * self.repeat
        else:
            return self.size