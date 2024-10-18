'''
This script is not essential for training.
Just for reference when comparing audio data with eyes.
'''

import os
import torchaudio
import matplotlib.pyplot as plt

from DataLoader import dataloader


def show_waveform (wf, sr) :
    
    #new_sr = sr / 10
    new_sr = sr
    transformed_wf = torchaudio.transforms.Resample(sr, new_sr)(wf[0, :].view(1, -1))
    
    plt.figure()
    plt.plot(transformed_wf[0,:].numpy())
                
    return ;


def show_spectrogram (wf) :
    
    spectrogram = torchaudio.transforms.Spectrogram()(wf)
    
    plt.figure()

    plt.imshow(spectrogram.log2()[0, :, :].numpy(), cmap = "viridis")

    return ;


def show_mfcc (wf, sr) :
    
    mfcc = torchaudio.transforms.MFCC(sample_rate = sr)(wf)
    
    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc.log2()[0, :, :].numpy(), cmap = "viridis")
    
    plt.figure()
    plt.plot(mfcc.log2()[0,:,:].numpy())
    plt.draw()

    return ;


if (__name__ == "__main__") :

    file_path = ".ogg"
    wf, sr = torchaudio.load(file_path)

    show_waveform(wf, sr)
    show_spectrogram(wf)
    show_mfcc(wf, sr)
    
