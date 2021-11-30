# -*- coding: utf-8 -*-
'''
Filename : TensorDevice.py
Function : Package Parameter，Variable，Tensor for easy use of cuda
'''

import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
mySeed = np.random.RandomState(2019)

def LongTensorDevice(Tensor,use_cuda):
    if use_cuda:
        return torch.LongTensor(Tensor).cuda()
    else:
        return torch.LongTensor(Tensor)
    
def FloatTensorDevice(Tensor,use_cuda):
    if use_cuda:
        return torch.FloatTensor(Tensor).cuda()
    else:
        return torch.FloatTensor(Tensor)

def VariableDevice (Tensor,requires_grad,use_cuda):
    if use_cuda:
        return Variable(torch.FloatTensor(Tensor).cuda(),requires_grad =requires_grad)
    else:
        return Variable(torch.FloatTensor(Tensor),requires_grad =requires_grad)

def ParameterDevice (Tensor,requires_grad,use_cuda):
    if use_cuda:
        return nn.Parameter(torch.FloatTensor(Tensor).cuda(),requires_grad =requires_grad)
    else:
        return nn.Parameter(torch.FloatTensor(Tensor),requires_grad =requires_grad)
