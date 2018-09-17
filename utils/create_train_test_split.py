#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""


import glob
import os
import random
from shutil import move


random.seed(2018)
Full_data = glob.glob('/home/krishna/DIgit_recognition_CNN/CNN_FilterBank/Preprocessed_data/training/*.wav')
num_files = range(len(Full_data))
random.shuffle(num_files)

#### 80% training
create_train_dir = os.getcwd()+'/training/'
if not os.path.exists(create_train_dir):
    os.makedirs(create_train_dir)

create_test_dir = os.getcwd()+'/testing/'
if not os.path.exists(create_test_dir):
    os.makedirs(create_test_dir)



tot_train_files = int(len(Full_data)*0.8)
for el in num_files[1:tot_train_files]:
    source_path=Full_data[el]
    dest_path = create_train_dir+source_path.split('/')[-1]
    move(source_path,dest_path)
    

    
#tot_train_files = int(len(Full_data)*0.8)
for el in num_files[tot_train_files:len(Full_data)]:
    source_path=Full_data[el]
    dest_path = create_test_dir+source_path.split('/')[-1]
    move(source_path,dest_path)
    
    
######################################################
##### Create train and test list
fid_train = open('data_utils/train_list.txt','w')
fid_test  = open('data_utils/test_list.txt','w')
all_train = glob.glob(os.getcwd()+'/training/*.wav')
all_test = glob.glob(os.getcwd()+'/testing/*.wav')
for filepath in all_train:
    fid_train.write(filepath+'\n')
fid_train.close()
for filepath in all_test:
    fid_test.write(filepath+'\n')
fid_test.close()



