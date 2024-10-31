# Speaker Segmentation

Date : June 18 <br>
Writer : 장영주

```python
import pandas as pd
import librosa
import numpy as np
import os
import joblib
import noisereduce as nr
import matplotlib.pyplot as plt
import soundfile as sf

# 잡음 제거
def noise_reduction(input_path, output_path):
    # 오디오 파일 로드
    data, sample_rate = librosa.load(input_path, sr=16000)
    
    # 잡음 제거
    reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)
    
    # 잡음 제거된 파일 저장
    sf.write(output_path, reduced_noise, sample_rate)
    
    return reduced_noise, sample_rate

# 입력 파일 경로
input_file = 'test_audio.ogg'

# 출력 파일 경로 설정 (현재 디렉토리에 저장됨)
output_file = 'reduced_' + os.path.basename(input_file)

# 잡음 제거 함수 호출
reduced_audio, sample_rate = noise_reduction(input_file, output_file)

print(f"잡음 제거가 완료되었습니다. 잡음 제거된 파일: {output_file}")
```