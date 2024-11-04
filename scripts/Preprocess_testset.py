'''
This script is preprocessing testset.
Extract segments from test data by cutting off.
Many output will be generated, but the will be predicted and then summed in the "Model_Predictor.py"

Step 1. Load test data.
Step 2. Extract segment.
Step 3. Save.
'''

import os
import torchaudio
import matplotlib.pyplot as plt

from DataLoader import dataloader
from Image_Spectrogram import image_spectrogram


def preprocess_test_audio (input_path, output_path_base) :
    
    
    sample_rate = 48000
    duration = 0.5 
    # segment size = sample_rate * duration 

    y, sr = librosa.load(input_path, sr = sample_rate)
    
    save_segment(y, output_path_base)


def save_segment (data, base, i = 1) :
    
    if len(data) < 16800 : return i
    
    max_index = np.argmax(np.abs(data))

    start_index = max_index
    end_index = max_index + 24000 # 48000*0.5
    
    if end_index > len(data) :
        padding = end_index - len(data)
        data = np.pad(data, (0, padding), "constant")
        
    segment = data[start_index : end_index]
    
    
    sf.write(base + f"_{i}.ogg", segment, 48000)
    i += 1
    
    if (start_index > 9600) : start_index -= 9600
        
    i = save_segment(data[:start_index], base, i)
    i = save_segment(data[end_index :], base, i)

    return i


if (__name__ == "__main__") :

    input_path = "../data/sample/test"
    output_path = "../data/preprocessed/separated/test"
    
    for file_name in os.listdir(input_path) :
        if file_name.endswith(".ogg") : 
            file_path = os.path.join(input_path, file_name)
            file_id = os.path.splitext(file_name)
            output_base = os.path.join(output_path, file_id[0])
            
            preprocess_test_audio (file_path, output_base)

    
    test_data_loader = dataloader(output_path, test)

    image_spectrogram (test_data_loader, test)

