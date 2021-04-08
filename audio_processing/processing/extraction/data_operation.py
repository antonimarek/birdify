import numpy as np
from pydub import AudioSegment
import os
from tqdm import tqdm
from numba import jit
from scipy.ndimage import interpolation, grey_erosion, grey_dilation
from scipy.signal import stft
import array


def resample_to_file(in_path: str, out_path: str, in_format: str = 'mp3', out_format: str = 'mp3', sr: int = 22050,
                     channels: int = 1, ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    AudioSegment.converter = ffmpeg_path
    sample = AudioSegment.from_file(in_path, in_format)
    sample = sample.set_frame_rate(sr)
    if channels:
        sample = sample.set_channels(channels)
    sample.export(out_path, out_format)
    pass


def resample_dir(dir_path_in: str, dir_path_out: str, in_format: str = 'mp3', out_format: str = 'mp3', sr: int = 22050,
                 channels: int = 1):
    f_list = [f for f in os.listdir(dir_path_in) if '.' in f]
    for _, file in enumerate(tqdm(f_list, desc="Progress")):
        in_path = dir_path_in + '/' + file
        out_path = dir_path_out + '/' + file
        resample_to_file(in_path=in_path, out_path=out_path, in_format=in_format, out_format=out_format, sr=sr,
                         channels=channels)
    return f'{len(f_list)} files from {dir_path_in} resampled into {dir_path_out}'


@jit(nopython=True)
def median_clipping_kernel(matrix, row_median, col_median, multi):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i, j] > multi * row_median[i]) & (matrix[i, j] > multi * col_median[j]):
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return matrix


def median_clipping(matrix, multi):
    r_median = np.median(matrix, axis=1)
    c_median = np.median(matrix, axis=0)
    matrix = median_clipping_kernel(matrix, r_median, c_median, multi)
    return matrix


def binary_h_vector(matrix):
    bin_vec = np.zeros_like(matrix[0, :], dtype=int)
    for j in range(matrix.shape[1]):
        if any(matrix[:, j] == 1):
            bin_vec[j] = 1
        else:
            bin_vec[j] = 0
    return bin_vec


def extract_signal_noise(in_path, in_format, median_multi: float = 3, flip_mask: bool = True, sr: int = 22050):
    song = AudioSegment.from_file(in_path, in_format)
    samples = song.get_array_of_samples()
    samples = np.array(samples, dtype=float)

    window_size: int = 512
    overlapping = .75
    bot_bins_out: int = 4
    top_bins_out: int = 24

    filter_size = (4, 4)
    vector_filter = 4
    n_dilation = 3

    f, t, Zxx = stft(samples, nperseg=window_size, nfft=window_size, window='hann',
                     noverlap=int(window_size * overlapping),
                     fs=sr)

    # scaling
    Zxx_abs = np.abs(Zxx)
    Zxx = (Zxx_abs - Zxx_abs.min()) / (Zxx_abs.max() - Zxx_abs.min())

    Zxx = Zxx[bot_bins_out:-top_bins_out, :]

    Zxx = median_clipping(Zxx, median_multi)

    Zxx = grey_erosion(Zxx, size=filter_size)
    Zxx = grey_dilation(Zxx, size=filter_size)

    bin_vector = binary_h_vector(Zxx)
    for n in range(n_dilation):
        bin_vector = grey_dilation(bin_vector, size=vector_filter)

    masking = interpolation.zoom(bin_vector, zoom=len(samples) / len(bin_vector))
    masking = np.ma.make_mask(masking)

    if flip_mask:
        masking = ~masking

    signal_sample = np.ma.compressed(np.ma.masked_array(samples, masking, dtype=int))

    samples_array = array.array(song.array_type, signal_sample)
    new_sample = song._spawn(samples_array)
    return new_sample


def split_signal_noise(in_path: str, s_out_path: str, n_out_path: str, in_format: str = 'mp3', out_format: str = 'mp3',
                       sr: int = 22050,
                       ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    AudioSegment.converter = ffmpeg_path
    signal = extract_signal_noise(in_path, in_format, median_multi=3, flip_mask=True, sr=sr)
    noise = extract_signal_noise(in_path, in_format, median_multi=2.5, flip_mask=False, sr=sr)

    signal.export(s_out_path, out_format)
    noise.export(n_out_path, out_format)
    pass


def split_signal_noise_dir(dir_path_in: str, dir_path_out: str, in_format: str = 'mp3', out_format: str = 'mp3',
                           sr: int = 22050):
    f_list = [f for f in os.listdir(dir_path_in) if '.' in f]
    signal_path = dir_path_out + '/signal'
    noise_path = dir_path_out + '/noise'
    if not os.path.exists(signal_path):
        os.makedirs(signal_path)
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)
    for _, file in enumerate(tqdm(f_list, desc="Progress")):
        in_path = dir_path_in + '/' + file
        s_out_path = signal_path + '/' + file
        n_out_path = noise_path + '/' + file
        split_signal_noise(in_path, s_out_path, n_out_path, in_format=in_format, out_format=out_format, sr=sr)
    return f'{len(f_list)} files from {dir_path_in} signal/noise split into {dir_path_out}'
