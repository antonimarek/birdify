from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
import librosa
from numba import jit
from typing import List


@jit(nopython=True)
def min_max_scale(array):
    array = np.subtract(array, np.min(array))
    array = array / np.max(array)
    return array


def comprehensive_mfcc(in_path: str, deltas: bool = True, in_format: str = 'mp3', sr: int = 22050, f_scale: bool = True,
                       ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    AudioSegment.converter = ffmpeg_path
    sample = AudioSegment.from_file(in_path, in_format)
    sample = sample.get_array_of_samples()
    sample = np.array(sample, dtype=float)
    mfccs = librosa.feature.mfcc(sample, n_mfcc=13, sr=sr, hop_length=int(512 * .75), n_mels=128, n_fft=512)
    if f_scale:
        mfccs = min_max_scale(mfccs)
    if deltas:
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        if f_scale:
            delta_mfccs = min_max_scale(delta_mfccs)
            delta2_mfccs = min_max_scale(delta2_mfccs)
        comp_mfcc = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        return comp_mfcc
    return mfccs


def mel_spectrogram():
    pass


def feature_extraction(files: List[str], classes: List[int], dir_path: str, leftover: bool = False,
                       deltas: bool = True, feat_type: str = 'mfcc', s_len: int = 115, f_scale: bool = True,
                       in_format: str = 'mp3',
                       ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    """
    Function extracting audio features for machine learning.

    :param files: list of file names
    :param classes: list of classes (after factorization)
    :param dir_path: path to sample directory
    :param leftover: if True returns chunk cutting remains (can be input for augmentation)
    :param deltas: bool = True, calculate deltas and delta deltas
    :param feat_type: name of feature types ('mfcc', 'mel_spec')
    :param s_len: chunk length
    :param f_scale: if True min_max scaling features
    :param in_format: audio file format (default = 'mp3')
    :param ffmpeg_path: path to ffmpeg codec (necessary when in_format='mp3'
    :return: X, y, (X_leftover, y_leftover) - numpy.array array of features and classes (respectively)
    """
    X = []
    y = []
    X_lo = []
    y_lo = []
    for file, cl in tqdm(zip(files, classes), total=len(files)):
        if feat_type == 'mfcc':
            com_mfcc = comprehensive_mfcc(in_path=dir_path + file, in_format=in_format, ffmpeg_path=ffmpeg_path,
                                          deltas=deltas, f_scale = f_scale)
            subs_num = com_mfcc.shape[1] // s_len
            subs_lo = com_mfcc.shape[1] % s_len
            if subs_num > 0:
                for n in range(subs_num):
                    X.append(com_mfcc[:, n * s_len:(n + 1) * s_len])
                    y.append(cl)
            if leftover & (subs_lo > 0):
                X_lo.append(com_mfcc[:, -subs_lo:])
                y_lo.append(cl)
    X, y = np.array(X), np.array(y)
    if leftover:
        return X, y, X_lo, y_lo
    else:
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
