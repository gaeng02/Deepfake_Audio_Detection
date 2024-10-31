# Speaker Segmentation

Date : June 18 <br>
Writer : 장영주

```python
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
from pyannote.core import notebook
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os

# secret_token = ""
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token = secret_token)

# inference on the whole file
pipeline("test_audio.ogg")

# dump the diarization output to disk using RTTM format
with open("test_audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

# 전체 파일에 대한 화자 분리 수행
diarization = pipeline('test_audio.ogg')

# 결과를 파일로 저장
output_file = "test_audio.txt"

with open(output_file, "w") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        f.write(f"Speaker {speaker}: start={turn.start}, end={turn.end}\n")

print(f"Diarization results saved to {output_file}")

```

```python
from pyannote.audio import Model

# secret_token = ""
model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token = secret_token)

AUDIO_FILE = "test_audio.ogg"
REFERENCE = "test_audio.rttm"

from pyannote.database.util import load_rttm

reference = load_rttm(REFERENCE)["test_audio"]
reference
```