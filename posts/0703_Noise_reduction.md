# Noise Reduction

Date : June 03 <br>
Writer : 양경식

train에 있는 ogg 파일을 잡음 제거한 후 다시 ogg 파일로 변환하여 저장 <br>

**미리 “denoised” 폴더를 만들어야함.**


```python
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def noise_reduction (input_path, output_path) :
    data, rate = librosa.load(input_path, sr = 48000)
    
    # stft = Short Time Fourier Transform
    spectrogram = np.abs(librosa.stft(data)) 
    plt.figure(figsize = (16, 6))
    librosa.display.waveshow(y = data,sr = rate)
    plt.show()
 
    
    mask = np.mean(spectrogram, axis=1) < 0.01
    
    denoise = spectrogram[:]
    denoise[mask] = 0
    
    denoise = librosa.istft(denoise)
    sf.write(output_path, denoise, rate)
    
    plt.figure(figsize = (16, 6))
    librosa.display.waveshow(y = denoise,sr = rate)
    plt.show()

'''
start_path = f"{start_file}.ogg"
end_path = f"{end_file}.ogg"
noise_reduction (start_path, end_path)
'''
```