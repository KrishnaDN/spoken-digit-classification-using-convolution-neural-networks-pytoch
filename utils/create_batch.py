#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""


import numpy as np
import torch
import scipy.io.wavfile as wav
from torch.autograd import Variable

def get_label(filename):
    class_ID = filename.split('/')[-1].split('_')[0]
    if class_ID=='O':
        return 0
    elif class_ID=='Z':
        return 10
    else:
        return int(class_ID)

def create_batch(batch_list,window_size):
    batch_size=len(batch_list)
    sig_batch=np.zeros([batch_size,window_size])
    lab_batch=np.zeros(batch_size)
    for i in range(len(batch_list)):
        wavefile =batch_list[i]
        fs,signal = wav.read(wavefile)
        label=get_label(wavefile)
        sig_batch[i,:] = signal
        lab_batch[i] = label
    if torch.cuda.is_available():
        inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
        lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
    else:
        inp=Variable(torch.from_numpy(sig_batch).float())
        lab=Variable(torch.from_numpy(lab_batch).float())
    return inp,lab
