'''
This script helps load, process, and manage data efficiently.
Plus, it provides processing data functionally.

dataloader    = [dataset, batch_size, shuffle, num_workers]
# datase      = set of data which consists of [waveform, sample_rate, label, index]
# batch_size  = the number of data sample in each iteration
# shuffle     = whether to shuffle or not
# num_workers = CPU threads in parallel
'''

'''
torch
Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.
'''

import os
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def dataloader (path, label, batch = 2) :

    dataset = []
    
    for file_name in os.listdir(path) :
        if file_name.endswith(".ogg") : 
            file_path = os.path.join(path, file_name)
            wf, sr = torchaudio.load(file_path)
            file_id = os.path.splitext(file_name)
            
            dataset.append([wf, sr, label, file_id[0]])
            
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch, shuffle = True, num_workers = 0)
    
    return loader
