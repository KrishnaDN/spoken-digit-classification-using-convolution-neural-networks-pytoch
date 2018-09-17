
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""



import os
import glob
import soundfile as sf
import numpy as np



def compute_dur(files):
    durs=[]
    for wavfile in files:
        f = sf.SoundFile(wavfile)
        durs.append(len(f)/float(f.samplerate)*1000)
    maximum_dur=max(durs)
    mean_dur = np.mean(np.asarray(durs))
    return maximum_dur,mean_dur




def get_max_dur_train():
    global_max=[]
    dir_train=glob.glob('/home/krishna/DIgit_recognition_CNN/CNN_FilterBank/Data/training/*/')
    for digit_id  in dir_train:
        all_files = glob.glob(digit_id+'/*.wav')
        max_dur,mean_dur  = compute_dur(all_files)
        print('Folder  '+digit_id+' contains max dur '+str(max_dur)+'ms')
        print('Folder  '+digit_id+' contains avg dur '+str(mean_dur)+'ms')
        global_max.append(max_dur)
    #print(global_max)
        
    return np.max(np.asarray(global_max))

