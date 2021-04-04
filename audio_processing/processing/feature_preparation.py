from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
import librosa
from typing import List


def comprehensive_mfcc(in_path: str, in_format: str = 'mp3', sr: int = 22050):
    sample = AudioSegment.from_file(in_path, in_format)
    sample = sample.get_array_of_samples()
    sample = np.array(sample, dtype=float)
    mfccs = librosa.feature.mfcc(sample, n_mfcc=13, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    comp_mfcc = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    return comp_mfcc


def feature_extraction(files: List[str], classes: List[int], dir_path: str, s_len: int = 130, in_format: str = 'mp3'):
    X = []
    y = []
    for file, cl in tqdm(zip(files, classes), total=len(files)):
        com_mfcc = comprehensive_mfcc(in_path=dir_path + file, in_format=in_format)
        subs_num = com_mfcc.shape[1] // s_len
        for n in range(subs_num):
            X.append(com_mfcc[:, n*130:(n+1)*130])
            y.append(cl)
    X, y = np.array(X), np.array(y)
    return X, y


def class_balancing(X, y):
    pass


def normalize():
    pass


def time_shift():
    pass


def pitch_shift():
    pass


def combine_audio():
    pass


def add_noise():
    pass


def augmentation():
    pass
