from pydub import AudioSegment
import librosa
import sox
import pandas as pd
import numpy as np
from python_speech_features import mfcc, logfbank


def add_sample_info(df, path_column: str, s_rate: bool = True, n_samples: bool = True, dur: bool = True):
    sample_info = pd.DataFrame()

    for f in df.index:
        if s_rate:
            sample_info.loc[f, 'sample_rate'] = sox.file_info.sample_rate(df.loc[f, path_column])
        if n_samples:
            sample_info.loc[f, 'num_samples'] = sox.file_info.num_samples(df.loc[f, path_column])
        if dur:
            sample_info.loc[f, 'duration'] = sox.file_info.duration(df.loc[f, path_column])
    if s_rate:
        sample_info.sample_rate = sample_info.sample_rate.map(int)
    if n_samples:
        sample_info.num_samples = sample_info.num_samples.map(int)
    return sample_info


def fft_calc(y, rate: int):
    """
    Calculate fft.
    :param y: signal as numpy array
    :param rate: sampling rate as int
    :return: (magnitude, frequency values)
    """
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    magnitude = abs(np.fft.rfft(y) / n)
    return magnitude, freq
