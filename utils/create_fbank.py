#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""




import speechpy
from speechpy.feature import lmfe
import scipy.io.wavfile as wav
import numpy as np
import torch
from torch.autograd import Variable
##### Log mel filter bank energy extraction and computes CMVN


def Compute_filterbank(audio_file, frame_length=0.025,frame_stride=0.01):
    fs,audio_data = wav.read(audio_file)
    filterbank_energy = lmfe(audio_data,fs,frame_length,frame_stride,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    #filterbank_energy_inv = np.transpose(filterbank_energy)
    normalized =  speechpy.feature.processing.cmvn(filterbank_energy, variance_normalization=True)
    return np.transpose(normalized)

def get_label(filename):
    class_ID = filename.split('/')[-1].split('_')[0]
    if class_ID=='O':
        return 0
    elif class_ID=='Z':
        return 10
    else:
        return int(class_ID)

def compute_batch_feats(batch_list,batch_size,num_filterbanks=40):
    feat_matrix=[]
    label_matrix=[]
    for i in range(len(batch_list)):
        label = get_label(batch_list[i])
        label_matrix.append(label)
        feat = Compute_filterbank(batch_list[i])
        feat_matrix.append(feat)
    if torch.cuda.is_available():
        inp=Variable(torch.from_numpy(np.asarray(feat_matrix)).float().cuda().contiguous())
        lab=Variable(torch.from_numpy(np.asarray(label_matrix)).float().cuda().contiguous())
    else:
        inp=Variable(torch.from_numpy(np.asarray(feat_matrix)).float())
        lab=Variable(torch.from_numpy(np.asarray(label_matrix)).float())
    return inp,lab


