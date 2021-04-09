import numpy as np
from pydub import AudioSegment
from typing import List
from tqdm import tqdm
from .basics import *
import os


def audio_chunks(in_path: str, in_format: str, length: float, min_len_ratio: float = 1 / 3):
    """
    Cuts audio signal into chunks of equal size. Last chunk, if doesn't match length, will be padded, except if
    shorter than given length (specified by ratio).

    :param min_len_ratio: Min ratio of chunk length to full length to be padded
    :param length: Length of each chunk in seconds
    :param in_path: Input file path
    :param in_format: Input format
    :return: List of chunks (as numpy.array)
    """
    signal, srt = audio_to_array(in_path, in_format, sr=True)
    len_sample = int(srt * length)
    n_chunks = int(np.ceil(len(signal) / len_sample))
    chunk_list = []
    for n in range(n_chunks):
        chunk = signal[n * len_sample:(n + 1) * len_sample]
        if chunk.shape[-1] > min_len_ratio * len_sample:
            if chunk.shape[-1] < len_sample:
                chunk = padding(chunk, (len_sample,))
            chunk_list.append(chunk)
    chunk_list = np.array(chunk_list)
    return chunk_list


def save_in_format(element, f_name: str, ext: str, out_dir: str):
    if ext == 'npy':
        np.save(out_dir + f_name, element)
    elif ext == 'npz':
        np.savez_compressed(out_dir + f_name, a=element)
    else:
        # TODO
        # saving as other formats (mp3, wav, etc.)
        pass


def save_chunks(in_path: str, in_format: str, class_id: str, length: float, out_dir: str, out_format: str = 'npy',
                f_list=None, return_f_list: bool = False):
    """
    Saves chunked file in given format.

    :param return_f_list:
    :param class_id: Will function as first part of file name
    :param in_path: Input file path
    :param f_list: List of files, if None, then lists all files in input directory
    :param in_format: Format of input file
    :param length: Desired length of chunk (in seconds)
    :param out_format: Output format
    :param out_dir: Output directory to store chunks
    :return: If return_f_list, returns updated list of files in directory
    """
    if f_list is None:
        f_list = [f for f in os.listdir(out_dir) if '.' in f]

    chunks = audio_chunks(in_path=in_path, in_format=in_format, length=length)
    for n in chunks:
        chunk_name = uq_name(base_name=class_id, f_list=f_list, ext=out_format)
        f_list.append(chunk_name)
        save_in_format(n, chunk_name, out_format, out_dir)
    if return_f_list:
        return f_list
    pass


def serial_chunks(in_paths: List[str], in_format: str, class_ids: List[str], length: float, out_dir: str,
                  out_format: str = 'npy'):
    """
    Performs chunking for all files in list and saves results in specified directory.

    :param in_paths: List of input paths
    :param in_format: Format of input files
    :param class_ids: List of names specific for each file (eg. class name or number), will function as first
     part of file name
    :param length: Desired length of chunk (in seconds)
    :param out_dir: Output directory to store chunks
    :param out_format: Output format
    :return: Nothing
    """
    f_list = [f for f in os.listdir(out_dir) if '.' in f]
    for file, cl in tqdm(zip(in_paths, class_ids), total=len(in_paths)):
        params = {'in_path': file,
                  'in_format': in_format,
                  'class_id': cl,
                  'length': length,
                  'out_dir': out_dir,
                  'f_list': f_list,
                  'out_format': out_format,
                  'return_f_list': True}
        f_list = save_chunks(**params)
    pass
