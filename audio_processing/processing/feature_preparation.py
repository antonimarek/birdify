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


def feature_extraction(files: List[str], classes: List[str], dir_path: str):
    for file in tqdm(zip(files, classes), total=len(files)):
        com_mfcc = comprehensive_mfcc(in_path=dir_path+file)
        subs_num = com_mfcc.shape[1]//130
    pass


def class_balancing():
    pass


def augmentation():
    pass
