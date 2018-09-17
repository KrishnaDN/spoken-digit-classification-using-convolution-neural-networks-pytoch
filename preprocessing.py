#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""



import glob
import os
from utils.compute_dur import get_max_dur_train
import scipy.io.wavfile as wav
import numpy as np


sampling_rate = 8000
maximum_dur_ms = get_max_dur_train()
global_length = int(maximum_dur_ms/1000*sampling_rate)



def pad_zeros(wave_file,global_length):
    fs,audio_data = wav.read(wave_file)
    pad_len = global_length-len(audio_data)
    padded_data = np.append(audio_data,np.zeros((1,pad_len))[0])
    return padded_data




##### Appending zeros to make all the sentences same length
preprocessed_dir =  os.getcwd()+'/Preprocessed_data/training'
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

training_data = glob.glob(os.getcwd()+'/Data/training/*/')
for folder in training_data:
    all_files = glob.glob(folder+'/*.wav')
    for wave_file in all_files:
        padded_data = pad_zeros(wave_file,global_length)
        save_file = preprocessed_dir+'/'+wave_file.split('/')[-2]+'_'+wave_file.split('/')[-1]
        wav.write(save_file,sampling_rate,padded_data/np.max(padded_data))




