# Deepfake_Audio_Detection
1. [Purpose](#purpose)
2. [Develop setting](#develop-environment)
3. [Contest](#contest)
4. [License](#license)

## Purpose
Currently, **Deepfake** issues are emerging globally.  
While deepfake technology can be used for personal and even positive purposes, it also has the potential to spread into areas like phishing scams and political manipulation.   
A few years ago, attentive listening could help detect deepfakes, but today they have reached a level where it is nearly impossible for human senses to distinguish them.  

For these reasons, the purpose of this program is **analyzing audio data to determine whether a voice is genuinely human or generated**through deepfake technology."


## Develop Setting

### Environment
`Python 3.12`

### Architecture


### Directory
#### Data is not provided due to copyright.  
├─ data   
&emsp;&emsp;├─ sample   
&emsp;&emsp;&emsp;&emsp;├─ train.csv   
&emsp;&emsp;&emsp;&emsp;├─ train/   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└─ audio_00001.ogg, ...  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(audio_files for train)   
&emsp;&emsp;&emsp;&emsp;├─ test.csv    
&emsp;&emsp;&emsp;&emsp;└─ test/   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└─ audio_00001.wav, ...  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(audio_files for test)   
&emsp;&emsp;├─ separated  
&emsp;&emsp;&emsp;&emsp;├─ true  
&emsp;&emsp;&emsp;&emsp;└─ false  
&emsp;&emsp;├─ preprocessed  
&emsp;&emsp;&emsp;&emsp;├─ separated    
&emsp;&emsp;&emsp;&emsp;├─ normalized  
&emsp;&emsp;&emsp;&emsp;└─ spectrogram  
&emsp;&emsp;├─ test_data  
&emsp;&emsp;&emsp;&emsp;├─ normalized  
&emsp;&emsp;&emsp;&emsp;└─ spectrogram  
&emsp;&emsp;└─ model.pth   
└─ src    


## Contest
[Link](https://dacon.io/competitions/official/236253/overview/description)

### Member
| 팀장 | 팀원 | 팀원 | 팀원 |
| :---: | :---: | :---: | :---: |
| [양경식](https://github.com/gaeng02)| [장영주](https://github.com/youngju6143) | 고진 | 박성민 |

### Schedule
|   SUN   |   MON   |   TUE   |   WED   |   THU   |   FRI   |   SAT   |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  | 06.24 <br> [Meeting #1](./docs/posts/0624_Meeting.md)| 06.25 <br> | 06.26 | 06.27 | 06.28 <br> [Meeting #2](./docs/posts/0628_Meeting.md) | 06.29 |
| 06.30 | 07.01 <br> [Train_with_Waveform](./docs/posts/0701_Train_with_waveform.md) | 07.02 <br> [Waveform_extraciton](./docs/posts/0702_Waveform_extraction.md) | 07.03 <br> [Noise_reduction](./docs/posts/0703_Noise_reduction.md) | 07.04 <br> [Separation_directory](./docs/posts/0704_Separation_directory.md) | 07.05 | 07.06 |
| 07.07 <br> [Issue#1](./docs/posts/0707_Issue_1.md) <br> [Issue#2](./docs/posts/0707_Issue_2.md)| 07.08 <br> [Meeting #3](./docs/posts/0708_Meeting.md) | 07.09 | 07.10 | 07.11 | 07.12 | 07.13 |
| 07.14 | 07.15 <br> [Meeting #4](./docs/posts/0715_Meeting.md) | 07.16 | 07.17 | 07.18 <br> [Noise_reduction](./docs/posts/0718_Noise_reduction.md) <br> [Segmentation](./docs/posts/0718_Segmentation.md) <br> [Model_build](./docs/posts/0718_Model_build.md) | 07.19 (Deadline) <br> [No_voice_reduction](./docs/posts/0719_No_voice_reduction.md) | |


## License
#### This project is licensed under the `MIT License`. 

### Using Library (Refer to [NOTICE](./NOTICE))

| Library | Version | License | 
|---|:---:|:---|
| [`Noisereduce`](https://github.com/timsainb/noisereduce?tab=MIT-1-ov-file#readme) | 3.0.2 | `MIT` |
| [`Torchinfo`](https://github.com/tyleryep/torchinfo) | 1.8.0 | `MIT` |
| [`Pydub`](https://pydub.com/) |  | `MIT` |
| [`Soundfile`](https://github.com/bastibe/python-soundfile) | 0.12.1 | `BSD` |
| [`Pandas`](https://github.com/pandas-dev/pandas) | 2.2.2 | `BSD` |
| [`Numpy`](https://github.com/numpy/numpy) | 1.26.4 | `BSD` |
| [`Joblib`](https://joblib.readthedocs.io/en/stable/) | 1.4.2 | `BSD` |
| [`Scikit-learn`](https://scikit-learn.org/stable/) | 1.5.1 | `BSD` |
| [`Torch`](https://pytorch.org/) | 2.3.1 cpu | `BSD` |
| [`Torchvision`](https://github.com/pytorch/vision) |  | `BSD` |
| [`Torchaudio`](https://github.com/pytorch/audio) | 2.3.1 cpu | `BSD` |
| [`Matplotlib`](https://github.com/matplotlib/matplotlib) | 3.9.1 | `Python Software Foundation` |
| [`Librosa`](https://github.com/librosa/librosa) | 0.10.2.post1 | `ISC` |
| [`Tensorflow`](https://github.com/tensorflow) | 2.17.0 | `Apache 2.0` |
| [`PIL`](https://python-pillow.org) | 10.3.0 | `HPND` |
