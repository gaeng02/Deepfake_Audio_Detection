import pandas as pd
import os
import shutil

if (__name__ == "__main__") :

    csv_file = "./sample/train.csv"
    output_dir = "./sample/train"
    

    df = pd.read_csv(csv_file)
    os.makedirs(output, exist_ok=True)

    for idx, row in df.iterrows():
        file_path = os.path.join("open", row["path"])
        
        if os.path.exists(file_path) :
            dest_path = os.path.join(output, row["id"] + ".ogg")
            shutil.copy(file_path, dest_path)
            
        else :
            print(f"File not found: {file_path}")
