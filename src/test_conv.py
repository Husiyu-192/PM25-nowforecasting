import h5py
import torch
import numbers
import os,sys
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
# import tensorwatch as tw
import torchvision
from torchvision import transforms
from dataset import H5Dataset
# from torchsummary import summary
#from tensorboardX import SummaryWriter
# from conf import settings
# from utils import *
# import WarmUpLR
# from model_complexity import compute_model_complexity
# from modelsize_estimate import compute_modelsize
# from gpu_mem_track import  MemTracker
from airConvlstm import *
def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def test(model, device, test_loader, crit):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data = Variable(sample['input'].unsqueeze(-1).permute(0, 1, 4, 2, 3).float()).to(device)
            target = Variable(sample['output'].squeeze().float()).to(device)
            output = model(data).squeeze()
            test_loss += crit(output, target).item()
    test_loss /= len(test_loader)
    # Horovod: average metric values across workers.
    # test_loss = metric_average(test_loss, 'avg_loss')
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

headers=["pm25"]
test_path = "./test_daqisuo_PM25_1to1.h5"
h5test =H5Dataset(test_path)
loader_test = DataLoader(h5test, batch_size=1,shuffle=False,num_workers=16)

height = 339 #269
width = 432 #239
# input_dim = 10 #26
input_dim = 1
n_layer = 2
hidden_size = [64, 128]
output_dim = 1
n_epoch = 1000
learning_rate = 1e-4
weight_decay = 0.9
weight_decay_epoch = 10
MSEmetric = nn.MSELoss()

air = AirConvLSTM(input_size=(height, width),
                    input_dim=len(headers),
                    hidden_dim=hidden_size,
                    kernel_size=(3, 3),
                    num_layers=n_layer,
                    output_dim=output_dim,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)

# if torch.cuda.device_count() > 1:
#     air  = nn.DataParallel(air)
air.to(device)

path=r"/home/daqisuo_output/model/model_conv/new_epoch_19.pt"
checkpoint = torch.load(path)
air.load_state_dict(checkpoint['model'])

# final_model, optimizer =load_checkpoint(model, name, optimizer)
start = time.time()
test(air, device, loader_test, MSEmetric)
end = time.time()
time_cost = sec_to_hms(int(end-start))
print('test data  time costs = {}'.format(time_cost))
