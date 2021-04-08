from pydub import AudioSegment
from .basics import *


def audio_chunks(in_path: str, in_format: str, length: float, out_dir, out_format: str = None, as_npy: bool = False):
    """
    Cuts audio signal into chunks of equal size. Last chunk, if doesn't match length, will be padded, except if
    shorter than 1/3 of length.

    :param length: Length of each chunk in seconds
    :param in_path: Input file path
    :param in_format: Input format
    :param out_format: Output format
    :param out_dir: Output directory to store chunks
    :param as_npy: If True, samples stored as .npy
    :return: Nothing
    """
    signal, fps = audio_to_array(in_path, in_format, sr=True)

    if as_npy:
        # save sample to numpy
        pass
    else:
        # save to output format
        pass
    pass


def save_chunks():
    # saves chunks to some directory
    pass


def serial_chunks():
    # chunks files from whole directory and saves
    pass
