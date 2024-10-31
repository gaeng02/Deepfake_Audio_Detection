'''
This script normalizes audio files of different lengths and splits them into a consistent duration.

Step 1. Find the maximum value based on the absolute value.
Step 2. Cut the following part to the specified duration.
Step 3. If the length is insufficient, apply padding.
'''

import os
import numpy as np
import librosa
import soundfile as sf

def _normalize_audio (input_path, output_path, sample_rate, duration) : 
        
    data, sr = librosa.load(input_path, sr = sample_rate) 
    
    # Step 1. Find
    max_index = np.argmax(np.abs(data))
    
    # Step 2. Cut
    start_index = max_index
    end_index = start_index + duration * sample_rate
    
    # Step 3. Padding
    if end_index > len(data) : 
        padding = end_index - len(data)
        data = np.pad(data, (0, padding), "constant")
        
    segment = data[start_index : end_index]
    
    sf.write(output_path, segment, sample_rate)
  
def normalize (input_dir, output_dir, sample_rate, duration) :
    
    print(f"Start :: {input_dir}")
    
    for file_name in os.listdir (input_dir) : 
        if file_name.endswith(".ogg") : 
            file_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            _normalize_audio(file_path, output_path, sample_rate, duration)
            
    print(f"Done :: {output_dir}")

if (__name__ == "__main__") :

    # Adjust sample_rate, duration 
    # Cutting Size = sample_rate * duration
    sample_rate = 48000
    duration = 2 # second

    # true label data
    true_directory = "../data/preprocessed/true"
    normalized_true_directory = "../data/normalized/true"
    os.makedir(normalized_true_directory, exist_ok = True)
    normalize(true_directory, normalized_true_directory)

    # false label data
    false_directory = "../data/preprocessed/false"
    normalized_false_directory = "../data/normalized/false"
    os.makedir(normalized_false_directory, exist_ok = True)
    normalize(false_directory, normalized_false_directory)
