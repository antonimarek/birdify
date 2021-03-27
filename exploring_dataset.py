# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:26:21 2021

@author: amarek
"""

from python_speech_features import mfcc, logfbank
import librosa
import audio_processing as ap
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytimeparse.timeparse import timeparse
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'metadata.csv')
data['Name'] = data.Genus +' '+ data.Specific_epithet

data.set_index('Path', inplace=True)
data.Time = data.Time.where(data.Time.str.contains(':'))
data.Time = data.Time.apply(lambda x: timeparse(str(x)))

for f in data[data.Time.isna()].index:
    signal = AudioSegment.from_mp3(f)
    fs = signal.frame_rate
    signal, _ = librosa.load(f, sr=fs)
    data.at[f, 'Time'] = signal.shape[0]/fs

plotting = data.groupby(['Name'])['Time'].mean()

plt.figure(figsize=(15,5))
sns.barplot(x=plotting.index, y=plotting.values)
plt.xticks(rotation = 90)
plt.xlabel(None)
plt.ylabel('mean time [s]')
plt.title('Class distribution', fontdict={'size':16}, y=1.05)
plt.show()
