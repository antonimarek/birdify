from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
import librosa
from numba import jit
from scipy.signal import stft
from typing import List


@jit(nopython=True)
def min_max_scale(array):
    array = np.subtract(array, np.min(array))
    array = array / np.max(array)
    return array


def comprehensive_mfcc(in_path: str, deltas: bool = True, in_format: str = 'mp3', sr: int = 22050, f_scale: bool = True,
                       ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe", n_mel: int = 128):
    AudioSegment.converter = ffmpeg_path
    sample = AudioSegment.from_file(in_path, in_format)
    sample = sample.get_array_of_samples()
    sample = np.array(sample, dtype=float)
    mfccs = librosa.feature.mfcc(sample, n_mfcc=13, sr=sr, hop_length=int(512), n_mels=n_mel, n_fft=512)
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


def mel_spectrogram(in_path: str, in_format: str = 'mp3', sr: int = 22050, n_mel: int = 128,
                    ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    AudioSegment.converter = ffmpeg_path
    song = AudioSegment.from_file(in_path, in_format)
    samples = song.get_array_of_samples()
    samples = np.array(samples, dtype=float)

    window_size: int = 512
    overlapping = .5
    bot_bins_out: int = 5
    top_bins_out: int = 5
    mel_spec = librosa.feature.melspectrogram(samples, sr=sr, n_fft=window_size,
                                              hop_length=int(window_size * (1 - overlapping)),
                                              n_mels=n_mel)
    mel_spec = librosa.power_to_db(mel_spec)
    mel_spec = mel_spec[bot_bins_out:-top_bins_out]
    return mel_spec


def feature_extraction(files: List[str], classes: List[int], dir_path: str, leftover: bool = False, n_mel: int = 128,
                       to_dir: str = False, full_length: bool = True, padding: bool = True,
                       deltas: bool = True, feat_type: str = 'mel_spec', s_len: int = 256, f_scale: bool = True,
                       in_format: str = 'mp3',
                       ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    """
    Function extracting audio features for machine learning.

    :param padding:
    :param to_dir:
    :param full_length:
    :param n_mel: number of mel filters
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
                                          deltas=deltas, f_scale=f_scale, n_mel=n_mel)
            if not full_length:
                subs_num = com_mfcc.shape[1] // s_len
                subs_lo = com_mfcc.shape[1] % s_len
                i = 0
                if subs_num > 0:
                    for n in range(subs_num):
                        if not to_dir:
                            X.append(com_mfcc[:, n * s_len:(n + 1) * s_len])
                            y.append(cl)
                        else:
                            np.save(to_dir + 'c_' + str(cl) + '_' + file.split('.')[0] + str(i) + '.npy',
                                    com_mfcc[:, n * s_len:(n + 1) * s_len])
                            i += 1
                if leftover & (subs_lo > 0):
                    if padding:
                        com_mfcc_rest = com_mfcc[:, -subs_lo:]
                        cmf_s = com_mfcc_rest.shape
                        com_mfcc_rest = np.concatenate((com_mfcc_rest,
                                                        np.zeros((cmf_s[0], s_len - cmf_s[1]), dtype=int)), axis=1)
                    else:
                        com_mfcc_rest = com_mfcc[:, -subs_lo:]
                    if not to_dir:
                        X_lo.append(com_mfcc_rest)
                        y_lo.append(cl)
                    else:
                        np.save(to_dir + 'c_' + str(cl) + '_' + file.split('.')[0] + str(i) + '.npy',
                                com_mfcc_rest)
                        i += 1
            else:
                np.save(to_dir + 'c_' + str(cl) + '_' + file.split('.')[0] + '.npy', com_mfcc)
        elif feat_type == 'mel_spec':
            mel_spec = mel_spectrogram(in_path=dir_path + file, in_format=in_format, ffmpeg_path=ffmpeg_path,
                                       n_mel=n_mel)
            if not full_length:
                subs_num = mel_spec.shape[1] // s_len
                subs_lo = mel_spec.shape[1] % s_len
                i = 0
                if subs_num > 0:
                    for n in range(subs_num):
                        if not to_dir:
                            X.append(mel_spec[:, n * s_len:(n + 1) * s_len])
                            y.append(cl)
                        else:
                            np.save(to_dir + 'c_' + str(cl) + '_' + file.split('.')[0] + str(i) + '.npy',
                                    mel_spec[:, n * s_len:(n + 1) * s_len])
                            i += 1
                if leftover & (subs_lo > 0):
                    if padding:
                        mel_spec_rest = mel_spec[:, -subs_lo:]
                        msr_s = mel_spec_rest.shape
                        mel_spec_rest = np.concatenate((mel_spec_rest,
                                                        np.zeros((msr_s[0], s_len - msr_s[1]), dtype=int)), axis=1)
                    else:
                        mel_spec_rest = mel_spec[:, -subs_lo:]

                    if not to_dir:
                        X_lo.append(mel_spec_rest)
                        y_lo.append(cl)
                    else:
                        np.save(to_dir + 'c_' + str(cl) + '_' + file.split('.')[0] + str(i) + '.npy',
                                mel_spec_rest)
                        i += 1
            else:
                np.save(to_dir + 'c_' + str(cl) + '_' + file.split('.')[0] + '.npy', mel_spec)
    X, y = np.array(X), np.array(y)
    if not (to_dir or full_length):
        if leftover:
            return X, y, X_lo, y_lo
        else:
            return X, y
    else:
        pass

