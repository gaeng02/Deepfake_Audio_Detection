import pandas as pd
import shutil
import os

# os.getcwd()

df = pd.read_csv("./train.csv")

true_directory = "./true"
false_directory = "./false"

os.makedirs(true_directory, exist_ok = True)
os.makedirs(false_directory, exist_ok = True)

for index, row in df.iterrows() : 
    path = row["path"]
    label = row["label"]
    
    if label == "true" : destination = os.path.join(true_directory, os.path.basename(path))
    elif label == "false" : destination = os.path.join(false_dir, os.path.basename(path))
    else : print("error")
    
    shutil.move(path, destination)
