# Waveform extraction

Date : June 02 <br>
Writer : 양경식

“open/train.csv”의 데이터를 토대로 “image_path”에 저장

```python
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 

csv_file = "open/train.csv"
df = pd.read_csv(csv_file)

len(df) # the number of data

# save_path
image_path = "test_images"
os.makedirs(image_path, exist_ok=True)

def visualize (path, title, save) :
    data, rate = sf.read(path)
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(title)
    
    # plt.show()
    
    plt.savefig(save)
    plt.close()
    
    
for idx, row in df.iterrows() :
    file_path = "./open" + row["path"][1:]
    
    title = f"{row['id']}, {row['label']}"
    save_path = os.path.join(image_path, f"{row['id']}.png")
    visualize(file_path, title, save_path) 
```