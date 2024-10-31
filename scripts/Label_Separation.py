'''
This script processes the "../data/sample/train.csv" file, which contains the names of audio files and their labels ("true" or "false").
It separates the data based on these labels.
To preserve the original data, the audio data is stored in a separate directory.
'''

import pandas as pd
import shutil
import os

if (__name__ == "__main__") :

    # os.getcwd()

    df = pd.read_csv("../data/sample/train.csv")

    true_directory = "../data/processed/separated/true"
    false_directory = "../data/processed/separated/false"

    os.makedirs(true_directory, exist_ok = True)
    os.makedirs(false_directory, exist_ok = True)

    for index, row in df.iterrows() : 
        path = row["path"]
        label = row["label"]
        
        if label == "true" : destination = os.path.join(true_directory, os.path.basename(path))
        elif label == "false" : destination = os.path.join(false_directory, os.path.basename(path))
        else : print("error")
        
        shutil.copy2(path, destination)
        
