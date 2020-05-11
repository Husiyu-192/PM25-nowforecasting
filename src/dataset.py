import torch
import os
import random
import time
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from airConvlstm import *
import h5py

class H5Dataset(Dataset):
    def __init__(self,h5f_path,patch_size=0):
        super(H5Dataset, self).__init__()
        hfdata = h5py.File(h5f_path, 'r')
        self.data = hfdata.get('data')
        self.target = hfdata.get('label')
        self.patch_size=patch_size

    def __getitem__(self, index):
        dic =   {
                  'input': torch.from_numpy(self.data[index,...]).float(),
                  'output': torch.from_numpy(self.target[index,...]).float()
                }
        if not self.patch_size:
            sample=dic
        else:
            img_in,img_out,_=self.get_patch(dic[input],dic[output],self.patch_size)
            sample={'input':img_in,'output':img_out}
        return sample

    def __len__(self):
        return self.data.shape[0]
  
    def get_patch(self, img_in, img_tar, patch_size, ix=-1, iy=-1):
        (ih, iw) = img_in.shape[1:3] #(t,h,w,c)
        (th, tw) = (ih, iw)

        tp = patch_size
        ip = tp

        if ix == -1:
            ix = random.randrange(0, iw - ip + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip + 1)

        (tx, ty) = (ix, iy)

        img_in = img_in[:, iy:iy + ip, ix:ix + ip,:]
        img_tar = img_tar[:, ty:ty + tp, tx:tx + tp]
                
        info_patch = {
            'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

        return img_in, img_tar, info_patch      

class AirDataset(Dataset):

    def __init__(self, dataset, transform=None, target_transform=None, history=6):
        """
        """
        super(AirDataset, self).__init__()
        self.dataset = dataset
        self.idx_table = []
        self.transform = transform
        self.target_transform = target_transform
        self.history = history 
        
        kind = dataset.shape[3]
        self.ms = np.zeros((kind, 2))
        self.minmax = np.zeros((kind,2))
        for i in range(kind):
            all_data = dataset[:,:,:,i]
            average = np.average(all_data)
            std = np.std(all_data)
            self.ms[i,:] = (average, std)
            minval = np.min(all_data)
            maxval = np.max(all_data)
            self.minmax[i,:] = (minval,maxval)
            
        for t in range(self.history, dataset.shape[0]):
            self.idx_table.append(t)

    def __len__(self):
        return len(self.idx_table)

    def __getitem__(self, idx):
        t = self.idx_table[idx]
        # weather, morphology history data, one step ahead
        weather = self.dataset[t-self.history+1: t+1, :, :, 15:]
        # aqi history data, one step behind
        aqi = self.dataset[t-self.history:t, :, :, :15]

        input_data = np.concatenate((aqi, weather), -1)
        for i in range(input_data.shape[3]):
            all_data = input_data[:,:,:,i]
            all_data = (all_data - self.ms[i, 0]) / self.ms[i, 1]
            #all_data = (all_data - self.minmax[i, 0]) / (self.minmax[i, 1] - self.minmax[i,0])
            input_data[:,:,:,i] = all_data
            
#         for i in range(input_data.shape[3]):
#             print(np.average(input_data[:,:,:,i]), np.std(input_data[:,:,:,i]))

        # groud truth output data
        output_data = self.dataset[t, :, :, 0]
        #output_data = (output_data - self.minmax[0, 0]) / (self.minmax[0, 1] - self.minmax[0,0])

        if self.transform:
            input_data = self.transform(input_data)

        if self.target_transform:
            output_data = self.target_transform(output_data)

        sample =  {
                  'input': input_data.astype('float'),
                  'output': output_data.astype('float')
                  }
        return sample
