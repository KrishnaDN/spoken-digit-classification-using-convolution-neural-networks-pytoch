#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""



import os 
import glob
from shutil import copyfile


training_files = '/home/krishna/DIgit_recognition_CNN/dadabase english digits/wav/isolated_digits_ti_train_endpt'
testing_files = '/home/krishna/DIgit_recognition_CNN/dadabase english digits/wav/isolated_digits_ti_test_endpt'

Cur_dir = os.getcwd()
man_voices = glob.glob(training_files+'/MAN/*/')
women_voices = glob.glob(training_files+'/WOMAN/*/')
man_voices_test = glob.glob(testing_files+'/MAN/*/')
women_voices_test = glob.glob(testing_files+'/WOMAN/*/')


def extract_audio(audio_file,save_folder):
    filename= audio_file.split('/')[-1]
    folder_id = audio_file.split('/')[-2]
    class_id = filename.split('_')[0][0]
    create_folder = save_folder+'/'+class_id+'/'
    if not os.path.exists(create_folder):
        os.makedirs(create_folder)
    dest_file = create_folder+folder_id+'_'+filename
    copyfile(audio_file,dest_file)
    print(dest_file)


################ Training data

##### Extract men voices
save_folder=Cur_dir+'/Data/training/'
for man_voice in man_voices:
    wave_files = glob.glob(man_voice+'/*.wav')
    for audio_file in wave_files:
         extract_audio(audio_file,save_folder)
         print(audio_file)


##### Extract women voices
save_folder=Cur_dir+'/Data/training/'
for women_voice in women_voices:
    wave_files = glob.glob(women_voice+'/*.wav')
    for audio_file in wave_files:
         extract_audio(audio_file,save_folder)
         print(audio_file)

######################################
         
############## Testing data


##### Extract men voices
save_folder=Cur_dir+'/Data/training'
for man_voice in man_voices_test:
    wave_files = glob.glob(man_voice+'/*.wav')
    for audio_file in wave_files:
         extract_audio(audio_file,save_folder)
         print(audio_file)


##### Extract women voices
save_folder=Cur_dir+'/Data/training/'
for women_voice in women_voices_test:
    wave_files = glob.glob(women_voice+'/*.wav')
    for audio_file in wave_files:
         extract_audio(audio_file,save_folder)
         print(audio_file)




