import numpy as np
from pydub import AudioSegment
import uuid


def audio_to_array(in_path: str, in_format: str, sr: bool = False, as_sample: bool = False,
                   ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    """
    Converts audio signal from file to numpy array using AudioSegment library.

    :param as_sample: If True, returns AudioSegment sample
    :param sr: If True, returns sampling rate
    :param in_path: Path of audio file
    :param in_format: Format of audio file
    :param ffmpeg_path: Path of ffmpeg codec (important when decoding mp3)
    :return: Audio signal as numpy.array, if sr & a_type returns tuple(signal, sr, a_type), if only one True returns
    tuple(signal, additional element)
    """
    if in_format == 'mp3':
        AudioSegment.converter = ffmpeg_path
    sample = AudioSegment.from_file(in_path, in_format)
    song = sample
    srt = sample.frame_rate
    sample = sample.get_array_of_samples()
    sample = np.array(sample, dtype=float)
    if sr & as_sample:
        return sample, srt, song
    elif sr:
        return sample, srt
    elif as_sample:
        return sample, song
    else:
        return sample


def gen_name(base_name, add_length: int = 8):
    f_id = uuid.uuid4().hex.upper()[:add_length]
    return base_name + '_' + f_id


def uq_name(base_name: str, f_list: list, ext: str, add_length: int = 8):
    """
    Generates unique file name in directory.

    :param ext: File extension (as string, e.g. 'wav')
    :param base_name: Fixed name part
    :param f_list: List of files in directory
    :param add_length:
    :return:
    """
    new_name = gen_name(base_name=base_name, add_length=add_length) + '.' + ext
    while new_name in f_list:
        new_name = gen_name(base_name=base_name, add_length=add_length) + '.' + ext
    return new_name


def padding(sample, full_shape: tuple):
    """
    Provides padding for chunks/samples shorter then desired full length.

    :param sample: Chunk as numpy.array
    :param full_shape: Full shape of chunk
    :return: Zero padded chunk
    """
    act_len = sample.shape[-1]
    if len(full_shape) > 1:
        add_zeros = np.zeros((*full_shape[:-1], full_shape[-1] - act_len), dtype=int)
        sample = np.concatenate((sample, add_zeros), axis=1)
    else:
        add_zeros = np.zeros((full_shape[-1] - act_len))
        sample = np.concatenate((sample, add_zeros))
    return sample
