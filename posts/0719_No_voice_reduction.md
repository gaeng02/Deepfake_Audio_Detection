# No voice reduction

Date : June 19 <br>
Writer : 장영주

```python
from pyannote.core import notebook, Segment, Annotation
from pyannote.audio import Model

notebook.crop = Segment(0, 30)
user_token = ""

model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token = user_token)
```

```python
import pandas as pd
import librosa
import numpy as np
import os
import joblib
import noisereduce as nr
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio

from pyannote.core import notebook, Segment, Annotation
from pyannote.audio.pipelines import VoiceActivityDetection

# 잡음 제거
def noise_reduction(input_path, output_path):
  
    # 오디오 파일 로드
    data, sample_rate = librosa.load(input_path, sr = 16000)   

    # 잡음 제거
    reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)

    # 잡음 제거된 파일 저장
    sf.write(output_path, reduced_noise, sample_rate)

    return reduced_noise, sample_rate

# 이 아래부터 반복문 써서 테스트 데이터 하나씩 돌리면 될 것 같아요

# 입력 파일 경로
input_file = 'test_audio.ogg'

# 출력 파일 경로 설정 (현재 디렉토리에 저장됨)
output_file = 'reduced_' + os.path.basename(input_file)

# VAD 파이프라인 설정
pipeline = VoiceActivityDetection(segmentation=model)

# 하이퍼파라미터 설정
HYPER_PARAMETERS = {
    "min_duration_on": 0.0,    # 목소리가 있는 지속 시간 최소값
    "min_duration_off": 0.0    # 목소리가 없는 지속 시간 최소값
}

# 파이프라인 인스턴스화
pipeline.instantiate(HYPER_PARAMETERS)

# 오디오 파일 처리
vad = pipeline(input_file)
waveform, sample_rate = torchaudio.load(input_file)

# 잡음 제거 함수 호출
reduced_audio, sample_rate = noise_reduction(input_file, output_file)

# 목소리가 탐지되지 않는 경우
if len(vad) == 0:
    waveform = waveform * 0        
    # torchaudio.save 호출 시 format과 encoding 설정
    torchaudio.save(output_file, waveform, sample_rate, format='wav', encoding='PCM_F')
    print(f"Voice activity not detected in '{input_file}'. Muted.")      
else:
    print(f"Voice activity detected in '{input_file}'.")   
print(f"잡음 제거가 완료되었습니다. 잡음 제거된 파일: {output_file}")
```