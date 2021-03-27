from pydub import AudioSegment
import librosa
import sox
import pandas as pd
import numpy as np


def add_sample_info(df, path_column: str, s_rate: bool = True, n_samples: bool = True, dur: bool = True):
    sample_info = pd.DataFrame()

    for f in df.index:
        if s_rate:
            sample_info.loc[f, 'sample_rate'] = sox.file_info.sample_rate(df.loc[f, path_column])
        if n_samples:
            sample_info.loc[f, 'num_samples'] = sox.file_info.num_samples(df.loc[f, path_column])
        if dur:
            sample_info.loc[f, 'duration'] = sox.file_info.duration(df.loc[f, path_column])
    sample_info[['sample_rate', 'num_samples']] = sample_info[['sample_rate', 'num_samples']].applymap(int)
    return sample_info
