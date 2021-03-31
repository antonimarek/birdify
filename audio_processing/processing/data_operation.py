# import librosa
# import sox
import pandas as pd
import numpy as np
from pydub import AudioSegment
import os
# from python_speech_features import mfcc, logfbank


# def add_sample_info(df, path_column: str, s_rate: bool = True, n_samples: bool = True, dur: bool = True):
#     sample_info = pd.DataFrame()
#
#     for f in df.index:
#         if s_rate:
#             sample_info.loc[f, 'sample_rate'] = sox.file_info.sample_rate(df.loc[f, path_column])
#         if n_samples:
#             sample_info.loc[f, 'num_samples'] = sox.file_info.num_samples(df.loc[f, path_column])
#         if dur:
#             sample_info.loc[f, 'duration'] = sox.file_info.duration(df.loc[f, path_column])
#     if s_rate:
#         sample_info.sample_rate = sample_info.sample_rate.map(int)
#     if n_samples:
#         sample_info.num_samples = sample_info.num_samples.map(int)
#     return sample_info


def resample_to_file(in_path: str, out_path: str, in_format: str = 'mp3', out_format: str = 'mp3', sr: int = 22050):
    sample = AudioSegment.from_file(in_path, in_format)
    sample = sample.set_frame_rate(sr)
    sample.export(out_path, out_format)
    pass


def resample_dir(dir_path_in: str, dir_path_out: str, in_format: str = 'mp3', out_format: str = 'mp3', sr: int = 22050):
    f_list = os.listdir(dir_path_in)
    for file in f_list:
        in_path = dir_path_in + '/' + file
        out_path = dir_path_out + '/' + file
        resample_to_file(in_path=in_path, out_path=out_path, in_format=in_format, out_format=out_format, sr=sr)
    return f'{len(f_list)} files from {dir_path_in} resampled into {dir_path_out}'


# song = AudioSegment.from_mp3(r'./Sonus-naturalis-156938.mp3')
# samples = song.get_array_of_samples()
# samples = np.array(samples, dtype=float)


# def fft_calc(y, rate: int):
#     """
#     Calculate fft.
#     :param y: np.ndarray
#     :param rate: sampling rate as int
#     :return: (magnitude, frequency values)
#     """
#     n = len(y)
#     freq = np.fft.rfftfreq(n, d=1 / rate)
#     magnitude = abs(np.fft.rfft(y) / n)
#     return magnitude, freq
