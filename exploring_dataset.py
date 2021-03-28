# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:26:21 2021

@author: amarek
"""

from python_speech_features import mfcc, logfbank
import librosa
import audio_processing as ap
from audio_processing.processing import add_sample_info, fft_calc
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytimeparse.timeparse import timeparse
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta

data = pd.read_csv(r'metadata.csv')
data['Name'] = data.Genus +' '+ data.Specific_epithet

def timedeltas(x):
    t = datetime.strptime(x, '%M:%S').time()
    return timedelta(minutes=t.minute, seconds=t.second)

data['duration'] = data['Length'].apply(lambda x: timedeltas(x).total_seconds())

plotting = data.groupby(['Name'])['duration'].mean()

# plt.figure(figsize=(15,5))
# sns.barplot(x=plotting.index, y=plotting.values, saturation=0.6)
# plt.xticks(rotation = 90)
# plt.xlabel(None)
# plt.ylabel('mean time [s]')
# plt.title('Class distribution', fontdict={'size':16}, y=1.05)
# plt.show()

# data[['sample_rate', 'num_samples']] = add_sample_info(data, 'Path', dur=False)

# with open('dataset.pkl', 'wb') as f:
#     pickle.dump(data, f)

with open('dataset.pkl', 'rb') as f:
    data = pickle.load(f)

classes = data.Name.unique()

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes[:1]:
    ix = data[data.Name == c].index[-1]
    f = data.loc[ix].Path
    fs = data.loc[ix].sample_rate
    signal, rate = librosa.load(f, sr=fs)
    signals[c] = signal
    fft[c] = fft_calc(signal, rate)

# for c in class_names:
#     mp3_file = data[data.Name == c].iloc[0, 0]
#     song = AudioSegment.from_mp3(file)
#     fs = song.frame_rate
#     signal, _ = librosa.load(file, sr=fs)
#     signals[c] = signal
#     fft[c] = calc_fft(signal, fs)

#     bank = logfbank(signal[:fs], fs, nfilt=128, nfft=1200).T
#     fbank[c] = bank
#     mel = mfcc(signal[:fs], fs, numcep=64, nfilt=128, nfft=1200).T
#     mfccs[c] = mel
               