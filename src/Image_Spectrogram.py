'''
This script performs the task of converting audio data into images.
It transforms the "DataLoader" data structure into Spectrograms.
'''

import os
import torchaudio
import matplotlib.pyplot as plt

from DataLoader import dataloader


def image_spectrogram (data_loader, label) :
    
    directory = f"../data/preprocessed/spectrogram/{label}"

    if not os.path.isdir(directory) : print("Error"); return ;
    
    for i, data in enumerate(train_loader) : 

        wf = data[0]
        sr = data[1][0]
        label = data[2]
        ID = data[3]

        spectrogram_transform = torchaudio.transforms.Spectrogram()
        spectrogram_tensor = spectrogram_transform(wf)
            
        for idx in range (wf.size(0)) :  # iteration batch size
            log_spectrogram = spectrogram_tensor[idx].log2()[0].numpy()
                
            log_spectrogram = np.nan_to_num(log_spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
                
            fig = plt.figure()
            plt.imsave(f"{directory}/{ID[idx]}.png", log_spectrogram, cmap="viridis")
            plt.close(fig)


if (__name__ == "__main__") :
    dataloader()
