'''
This script is not essential for training.
Just for reference when deciding the size to set for cutting while normalization.

By checking the entire train data, find the largest and smallest audio files.
'''

import os
import librosa

def find_length () : 
    # Find Max, min size. 
    # return [Max, min]

    M, m = 0, 1237870
    path = ["../data/preprocessed/separated/true", "../data/preprocessed/separated/false"]
    
    for p in path : 
        for file_name in os.listdir(p) : 
            if file_name.endswith(".ogg") : 
                file_path = os.path.join(p, file_name)
                wf, sr = librosa.load(file_path, sr = 48000)

                M = max(M, len(wf))
                m = min(m, len(wf))
                
    return [M, m]

def find_object (m) :
    # Find audio file name of m size.
    # return audio file name

    path = ["../data/preprocessed/separated/true", "../data/preprocessed/separated/false"]

    for p in path : 
        for file_name in os.listdir(p) :
            if file_name.endswith(".ogg") : 
                    file_path = os.path.join(p, file_name)
                    wf, sr = librosa.load(file_path, sr = 48000)

                    if wf.shape[1] == m : print(file_path); return

    return None

if (__name__ == "__main__") :

    length = find_length() # [Max, Min]
    
    find_object(length[0]) # Max
    find_object(length[1]) # Min
