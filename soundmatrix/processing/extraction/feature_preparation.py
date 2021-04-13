# from pydub import AudioSegment
# from tqdm import tqdm
import numpy as np
# import librosa
from numba import jit
# from scipy.signal import stft
# from typing import List
from ..machine_learning.augmentation import normalize
import tensorflow_io as tfio
from tensorflow import transpose
from tensorflow import cast, float32


def get_feature(sample, f_type: str = 'mel_spec', max_len=None):
    if f_type == 'mel_spec':
        s_feat = tfio.experimental.audio.spectrogram(sample, nfft=1024, window=1024, stride=256)
        s_feat = cast(s_feat, dtype=float32)
        s_feat = tfio.experimental.audio.melscale(
            s_feat, rate=22050, mels=128, fmin=500, fmax=11000
        )
        s_feat = tfio.experimental.audio.dbscale(s_feat, top_db=80).numpy()
        s_feat = transpose(s_feat)
    # elif f_type == 'mpcc':
    #     s_feat = mfcc(sample)
    # elif f_type == 'mpcc_deltas':
    #     s_feat = mfcc(sample, deltas=True)
    else:
        raise ValueError('No feature type specified!')
    if max_len is not None:
        s_feat = s_feat[:, :max_len]
    s_feat = normalize(s_feat)
    f_shape = s_feat.shape
    s_feat = s_feat.reshape((*f_shape, 1))
    return s_feat


@jit(nopython=True)
def min_max_scale(array):
    array = np.subtract(array, np.min(array))
    array = array / np.max(array)
    return array


"""
def mfcc(sample, f_scale: bool = True, n_mel: int = 128, sr: int = 22050, deltas: bool = False):
    f_mfcc = librosa.feature.mfcc(sample, n_mfcc=13, sr=sr, hop_length=int(512), n_mels=n_mel, n_fft=512)
    if f_scale:
        f_mfcc = min_max_scale(f_mfcc)
    if deltas:
        delta_mfccs = librosa.feature.delta(f_mfcc, mode='nearest')
        delta2_mfccs = librosa.feature.delta(f_mfcc, order=2, mode='nearest')
        if f_scale:
            delta_mfccs = min_max_scale(delta_mfccs)
            delta2_mfccs = min_max_scale(delta2_mfccs)
        comp_mfcc = np.concatenate((f_mfcc, delta_mfccs, delta2_mfccs))
        return comp_mfcc
    return f_mfcc


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
        delta_mfccs = librosa.feature.delta(mfccs, mode='nearest')
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, mode='nearest')
        if f_scale:
            delta_mfccs = min_max_scale(delta_mfccs)
            delta2_mfccs = min_max_scale(delta2_mfccs)
        comp_mfcc = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        return comp_mfcc
    return mfccs


def mel_spec(sample, window_size: int = 512, overlapping=.5, bot_bins_out: int = 0, top_bins_out: int = 0,
             n_mel: int = 128, sr: int = 22050):
    m_spec = librosa.feature.melspectrogram(sample, sr=sr, n_fft=window_size,
                                            hop_length=int(window_size * (1 - overlapping)),
                                            n_mels=n_mel)
    m_spec = librosa.power_to_db(m_spec)
    m_spec = m_spec[bot_bins_out:-top_bins_out]
    return m_spec


def file_mel_spec(in_path: str, in_format: str = 'mp3', sr: int = 22050, n_mel: int = 128,
                  ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    AudioSegment.converter = ffmpeg_path
    song = AudioSegment.from_file(in_path, in_format)
    samples = song.get_array_of_samples()
    samples = np.array(samples, dtype=float)
    m_s = mel_spec(samples, sr=sr, n_mel=n_mel)
    return m_s


def feature_extraction(files: List[str], classes: List[int], dir_path: str, leftover: bool = False, n_mel: int = 128,
                       to_dir: str = False, full_length: bool = True, padding: bool = True,
                       deltas: bool = True, feat_type: str = 'mel_spec', s_len: int = 256, f_scale: bool = True,
                       in_format: str = 'mp3',
                       ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    # 
    # Function extracting audio features for machine learning.
    # 
    # :param padding:
    # :param to_dir:
    # :param full_length:
    # :param n_mel: number of mel filters
    # :param files: list of file names
    # :param classes: list of classes (after factorization)
    # :param dir_path: path to sample directory
    # :param leftover: if True returns chunk cutting remains (can be input for augmentation)
    # :param deltas: bool = True, calculate deltas and delta deltas
    # :param feat_type: name of feature types ('mfcc', 'mel_spec')
    # :param s_len: chunk length
    # :param f_scale: if True min_max scaling features
    # :param in_format: audio file format (default = 'mp3')
    # :param ffmpeg_path: path to ffmpeg codec (necessary when in_format='mp3'
    # :return: X, y, (X_leftover, y_leftover) - numpy.array array of features and classes (respectively)
    # 

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
            mel_spec = file_mel_spec(in_path=dir_path + file, in_format=in_format, ffmpeg_path=ffmpeg_path,
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
"""
