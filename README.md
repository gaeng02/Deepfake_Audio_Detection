# Deepfake_Audio_Detection
1. [Purpose](#purpose)
2. [Member]()
3. []()
4. []()
5. []()
6. []()

## Purpose

## Member



### Develop environment
- `Python 3.12`


### Schedule
|   SUN   |   MON   |   TUE   |   WED   |   THU   |   FRI   |   SAT   |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  | 06.24 <br> [Meeting #1](./posts/0624_Meeting.md)| 06.25 <br> | 06.26 | 06.27 | 06.28 <br> [Meeting #2](./posts/0628_Meeting.md) | 06.29 |
| 06.30 | 07.01 <br> [Train_with_Waveform](./posts/0701_Train_with_waveform.md) | 07.02 <br> [Waveform_extraciton](./posts/0702_Waveform_extraction.md) | 07.03 <br> [Noise_reduction](./posts/0703_Noise_reduction.md) | 07.04 <br> [Separation_directory](./posts/0704_Separation_directory.md) | 07.05 | 07.06 |
| 07.07 <br> [Issue#1](./posts/0707_Issue_1.md) <br> [Issue#2](./posts/0707_Issue_2.md)| 07.08 <br> [Meeting #3](./posts/0708_Meeting.md) | 07.09 | 07.10 | 07.11 | 07.12 | 07.13 |
| 07.14 | 07.15 <br> [Meeting #4](./posts/0715_Meeting.md) | 07.16 | 07.17 | 07.18 <br> [Noise_reduction](./posts/0718_Noise_reduction.md) <br> [Segmentation](./posts/0718_Segmentation.md) <br> [Model_build](./posts/0718_Model_build.md) | 07.19 (Deadline) <br> [No_voice_reduction](./posts/0719_No_voice_reduction.md) | |


### Directory
├─ data <br>
&emsp;&emsp;├─ train.csv <br>
&emsp;&emsp;├─ train/ <br>
&emsp;&emsp;&emsp;&emsp;└─ audio_00001.ogg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(audio_files for train)<br>
&emsp;&emsp;├─ test.csv <br>
&emsp;&emsp;└─ test/ <br>
&emsp;&emsp;&emsp;&emsp;└─ audio_00001.wav <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(audio_files for test)<br>
├─ preprocess_data <br>


### Library License
| Library | Version | License | 
|---|:---:|:---|
| [`Pandas`](https://github.com/pandas-dev/pandas) | 2.2.2 | `BSD` |
| [`Numpy`](https://github.com/numpy/numpy) | 1.26.4 | `BSD` |
| [`Matplotlib`](https://github.com/matplotlib/matplotlib) | 3.9.1 | `Python Software Foundation` |
| [`Soundfile`](https://github.com/bastibe/python-soundfile) | 0.12.1 | `BSD` |
| [`Librosa`](https://github.com/librosa/librosa) | 0.10.2.post1 | `ISC` |
| [`Pyannote`](https://github.com/pyannote) |  | `MIT` |
| [`Pydub`](https://pydub.com/) |  | `MIT` |
| [`Joblib`](https://joblib.readthedocs.io/en/stable/) |  | `BSD` |
| [`Noisereduce`](https://github.com/timsainb/noisereduce?tab=MIT-1-ov-file#readme) |  | `MIT` |
| [`Scikit-learn`](https://scikit-learn.org/stable/) | 1.5.1 | `BSD` |
| [`Torch`](https://pytorch.org/) | 2.3.1 cpu | `BSD` |
| [`Torchvision`](https://github.com/pytorch/vision) |  | `BSD` |
| [`Torchaudio`](https://github.com/pytorch/audio) | 2.3.1 cpu | `BSD` |
| [`Torchinfo`](https://github.com/tyleryep/torchinfo) | 1.8.0 | `MIT` |
| [`Tensorflow`](https://github.com/tensorflow) | 2.17.0 | `Apache 2.0` |