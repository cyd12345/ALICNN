# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:30:29 2021

@author: Excellent_FOX
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from PIL import Image
import collections
import time
from pathlib import Path

# path = Path(r"D:\UNET")
# os.chdir(path)

# from stacked import stackMatrix
from generatedata import *
from tool import *

CUDA = torch.cuda.is_available()

def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, **kwargs):
    device = 'cuda' if CUDA else 'cpu'
    return torch.as_tensor(x, device=device, **kwargs)

def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata-min)/(max-min)
    return outputdata

class Conv_block(nn.Module):
    def __init__(self,
             feeling_field_1=4,
             feeling_field_2=8,
             inhibition_field=16,
             inputchannel=128,
             outputchannel=128,
             kernel_size_1=3,
             kernel_size_2=3,
             kernel_size_3=5):
        super().__init__()
        self.feeling_1 = nn.Conv2d(inputchannel,outputchannel,kernel_size=kernel_size_1,stride=1,padding=1)
        self.relu_1 = nn.LeakyReLU()
        self.feeling_2 = nn.Conv2d(inputchannel,outputchannel,kernel_size=kernel_size_2,stride=1,padding=1)
        self.relu_2 = nn.LeakyReLU()
        self.inhibition = nn.Conv2d(inputchannel,outputchannel,kernel_size=kernel_size_3,stride=1,padding=2)
        self.relu_3 = nn.LeakyReLU()

        self.feeling_field_1 = feeling_field_1
        self.feeling_field_2 = feeling_field_2
        self.inhibition_field = inhibition_field
    
    def forward(self,x):
        #x:(N,C,W,H)
        #print('x',x.size())
        mid = x.size()[3]//2
        f1 = self.feeling_1(x[:,:,:,mid-self.feeling_field_1:mid+self.feeling_field_1])
        f1 = self.relu_1(f1)
        #print('f1',f1.size())
        f2 = self.feeling_2(x[:,:,:,mid-self.feeling_field_2:mid+self.feeling_field_2])
        f2 = self.relu_2(f2)
        #print('f2',f2.size())
        i = self.inhibition(x[:,:,:,mid-self.inhibition_field:mid+self.inhibition_field])
        i = self.relu_3(i)
        #print('i',i.size())
        y = torch.zeros(x.size())
        #print('y',y.size())
        y[:,:,:,mid-self.feeling_field_1:mid+self.feeling_field_1] =+ f1
        y[:,:,:,mid-self.feeling_field_2:mid+self.feeling_field_2] =+ f2
        y =- i
        #print('y',y.size())
        return y
    
class position(nn.Module):#用线性层生成位置图谱
    def __init__(self,nw=1024):
        super().__init__()
        self.position = nn.Linear(nw//2+1,nw//2+1)
        self.relu = nn.LeakyReLU()
    
    def forward(self,x):
        y = self.position(x)
        y = self.relu(y)
        y = y.transpose(2,1)
        return y

class CNN(nn.Module):#使用256*256线性层训练位置图谱
    def __init__(self,
                 win_len=32,
                 win_step=8,
                 width=1):
        super().__init__()
        self.position = position()
        self.encoder = nn.Conv2d(width+1, 128, 3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(128)
        #self.repeat =  self._Sequential_block(
            #3, in_channels=128, out_channels=128, kernel_size=3)
        self.conv1 = nn.Conv2d(129, 64, 3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.relu6 = nn.LeakyReLU()

        self.gen_mask = nn.Sequential(nn.Conv2d(64, 128, 1, stride=1),
                                      nn.Sigmoid())

        self.decoder = nn.ConvTranspose2d(128, 1, 1, stride=1)

        self.win_len = win_len
        self.win_step = win_step

        self.feature_names = ['conv1', 'relu1',
                              'conv2', 'relu2',
                              'conv3', 'relu3',
                              'conv4', 'relu4'
                              ]

        self.features = [self.conv1, self.relu1,
                         self.conv2, self.relu2,
                         self.conv3, self.relu3,
                         self.conv4, self.relu4,
                         self.conv5, self.relu5,
                         self.conv6, self.relu6
                         ]


    '''   
    def _Sequential_block(self, num_blocks, **block_kwargs):
        ''''''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        ''''''
        Conv1D_Block_lists = [Conv_block(
            **block_kwargs) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_Block_lists)
    '''

    def forward(self,x):
        p = self.position(x)  #linear layer
        p = p.transpose(2, 1)
        x = torch.stack((x, p), dim=1)  #Stitching matrix 'p'
        p = p.unsqueeze(1)
        x = self.encoder(x)
        x1 = self.norm(x)
        for name, layer in zip(self.feature_names, self.features):   #Concatenate the matrix 'p' before each convolution
            if 'conv' in name:
                x1 = torch.cat((p, x1), dim=1)
                x1 = layer(x1)
        m = self.gen_mask(x1)  #generate mask
        y = self.decoder(m*x).squeeze()
        return y

    def logits_and_activations(self, x, layer_names, as_dict=False, suppression_masks={}, save_maps=False,
                               without_last_relu=True):

        needed_layers = set(layer_names)
        if suppression_masks != {}:
            assert all(layer in suppression_masks for layer in needed_layers)
        layer_values = {}
        maps_to_print = collections.defaultdict(list)

        p = self.position(x)
        p = p.transpose(2, 1)
        x = torch.stack((x, p), dim=1)
        p = p.unsqueeze(1)
        x = self.encoder(x)
        x1 = self.norm(x)
        for name, layer in zip(self.feature_names, self.features):
            if 'conv' in name:
                x1 = torch.cat((p, x1), dim=1)

            x1 = layer(x1)

            if name in suppression_masks:
                if save_maps:
                    maps_to_print[name].append(to_np(suppression_masks[name].squeeze()))
                    maps_to_print[name].append(to_np(x1.squeeze(0).sum(0)))

                # applying suppression mask to relu layer
                if not (without_last_relu and name == 'relu5_3'):
                    sup_mask_sized_as_x = suppression_masks[name].expand(-1, x1.size()[1], -1, -1)
                    x1 = x1.where(sup_mask_sized_as_x != 0, torch.zeros_like(x1))

                if save_maps:
                    maps_to_print[name].append(to_np(x1.squeeze(0).sum(0)))

            if name in needed_layers:
                layer_values[name] = x1

        if not as_dict:
            layer_values = [layer_values[n] for n in layer_names]

        m = self.gen_mask(x1)
        y = self.decoder(x*m).squeeze()
        return y, layer_values, maps_to_print

    def predict(self, x):
        """
        Return predicted class IDs.
        """
        logits = self(x)
        _, prediction = logits.max(1)
        return prediction[0].item()


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_convtasnet():
    x = torch.randn(1,3,256,256)
    nnet = feel_inhibition_conv()
    s = nnet(x)
    print(str(check_parameters(nnet))+' Mb')
    print(s[1].shape)