'''
This script is reducing noise.

Before test the testset, preprocess by reducing noise audio data.
'''

import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import os

from Visualize import show_spectrogram 


def noise_reduction_nr (input_path, output_path) :
    
    data, sample_rate = librosa.load(input_path, sr = 48000)
    
    reduced_noise = nr.reduce_noise(y = data, sr = sample_rate)

    sf.write(output_path, reduced_noise, sample_rate)
    
    return reduced_noise, sample_rate


def noise_reduction_fft (input_path, output_path) :

    sample_rate = 48000
    cutoff_freq = 5000
    # audio size = 48000 * 5000

    data, sr = librosa.load(input_path, sr = sample_rate)
    
    D = librosa.stft(data)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    
    idx_cutoff = np.where(freqs >= cutoff_freq)[0][0]
    D[idx_cutoff:, :] = 0
    
    reduced_noise = librosa.istft(D)

    sf.write(output_path, reduced_noise, sr)
    
    return reduced_noise, sample_rate


if (__name__ == "__main__") :

    input_dir = "../data/preprocessed/normalized"
    output_dir = "../data/preprocessed/noise_reduced"

    for file_name in os.listdir (input_dir) : 
        if file_name.endswith(".ogg") : 
            file_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Choose 
            # noise_reduction_nr(file_path, output_path)
            # noise_reduction_fft(file_path, output_path)
    
    show_spectrogram()
