# Seperation real/fake label

Date : June 04 <br>
Writer : 양경식

train에 있는 ogg 파일을 train.csv에 근거하여 real, fake 구분하여 저장 <br>

**반드시 train의 복사를 만들어두고 이를 사용할 것. (원본 데이터 보장)** <br>
**code가 존재하는 같은 디렉토리에 real, fake 폴더를 만들어 둘 것**

```python
import pandas as pd
import shutil
import os

# os.getcwd()

df = pd.read_csv("./train.csv")

real_dir = "./real"
fake_dir = "./fake"

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

for index, row in df.iterrows() : 
    path = row["path"]
    label = row["label"]
    
    if label == "real" : destination = os.path.join(real_dir, os.path.basename(path))
    elif label == "fake" : destination = os.path.join(fake_dir, os.path.basename(path))
    else : print("error")
    
    shutil.move(path, destination)
    
```