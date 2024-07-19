import os
import numpy as np
import librosa
import soundfile as sf

def _preprocess_audio (input_path, output_path, sample_rate, duration) : 
        
    data, sr = librosa.load(input_path, sr = sample_rate) 
    
    max_index = np.argmax(np.abs(data))
    
    start_index = max_index
    end_index = start_index + duration * sample_rate
    
    # Padding
    if end_index > len(data) : 
        padding = end_index - len(data)
        data = np.pad(data, (0, padding), "constant")
        
    segment = data[start_index : end_index]
    
    sf.write(output_path, segment, sample_rate)

    
def preprocess (input_dir, output_dir, sample_rate, duration) :
    
    print(f"Start :: {input_dir}")
    
    for file_name in os.listdir (input_dir) : 
        if file_name.endswith(".ogg") : 
            file_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            _preprocess_audio(file_path, output_path, sample_rate, duration)
            
    print(f"Done :: {output_dir}")

    
if (__name__ == "__main__") :

    # Adjust sample_rate, duration 
    sample_rate = 48000
    duration = 2

    
    real_dir = "./real"
    fake_dir = "./fake"

    preprocess_real_dir = "./preprocess_real"
    preprocess_fake_dir = "./preprocess_fake"

    os.makedir(preprocess_real_dir, exist_ok = True)
    os.makedir(preprocess_fake_dir, exist_ok = True)

    preprocess (real_dir, preprocess_real_dir)
    preprocess (fake_dir, preprocess_fake_dir)

       
