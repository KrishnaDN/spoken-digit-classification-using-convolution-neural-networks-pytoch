#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Email:krishnadn94@gmail.com
@author: krishna
"""


import numpy as np
import torch
from models.FilterBank_ConvNet import ConvNet
import torch.nn as nn
import random
from utils.create_fbank import compute_batch_feats

random.seed(2018)


##### Data info
#max_dur_ms = get_max_dur_train()
sampling_rate=8000
#global_length= int((max_dur_ms/1000)*sampling_rate)


training_files =  [line.rstrip('\n') for line in open('data_utils/train_list.txt')]
testing_files = [line.rstrip('\n') for line in open('data_utils/test_list.txt')]
#batch_size=4
#batch_list = training_files[:batch_size]
#input_batch,labels = create_batch(batch_list,global_length)

### Hyper parameters
batch_size=32
learning_rate = 0.001
num_classes=11
N_epochs = 30
######
test_batch_size=16

#######

input_shape=[40,146]
###### Model configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ConvNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
seed=1234

#####
### Shuffling the list to randomize the data
random.shuffle(training_files)
######
N_batches = int(len(training_files)/batch_size)
N_batchs_test = int(len(testing_files)/test_batch_size)

for epoch in range(N_epochs):
    loss_sum=0
    err_sum=0
    acc_sum=0
    for i in range(N_batches):
        batch_list= training_files[i*batch_size:(i*batch_size)+batch_size]
        input_batch,input_labels = compute_batch_feats(batch_list,batch_size)
        inputs = torch.reshape(input_batch,(batch_size,1,input_shape[0],input_shape[1]))
        inputs = inputs.to(device)
        labels = input_labels.to(device) 
        output = model(inputs)
        loss = criterion(output,labels.long())
        prediction=torch.max(output,dim=1)[1]
        err = torch.mean((prediction!=labels.long()).float())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
        acc_sum  = acc_sum+torch.mean((prediction==labels.long()).float())
    loss_tot=loss_sum/N_batches
    err_tot=err_sum/N_batches
    tot_acc = acc_sum/N_batches
    print('Total training loss----->'+str(loss_tot))
    print('Total error loss----->'+str(err_tot))
    print('Total training accuracy----->'+str(tot_acc))
    
    ##### Testing accuracy after every epoch
    test_loss_sum=0
    test_err_sum=0
    test_acc_sum=0
    with torch.no_grad():
        for i in range(N_batchs_test):
            test_batch_list= testing_files[i*test_batch_size:(i*test_batch_size)+test_batch_size]
            input_batch,input_labels = compute_batch_feats(test_batch_list,batch_size)
            inputs = torch.reshape(input_batch,[batch_size,1,(batch_size,1,input_shape[0],input_shape[1])])
            inputs = inputs.to(device)
            labels = input_labels.to(device) 
            output = model(inputs)
            loss = criterion(output,labels.long())
            prediction=torch.max(output,dim=1)[1]
            test_acc_sum  = test_acc_sum+torch.mean((prediction==labels.long()).float())
        tot_acc = test_acc_sum/N_batchs_test
        print('Total testing accuracy----->'+str(tot_acc))
        
    
    
    
    
