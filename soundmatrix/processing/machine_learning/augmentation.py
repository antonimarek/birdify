import numpy as np
from soundmatrix.processing.extraction.feature_preparation import min_max_scale


def normalize(sample):
    sample = np.subtract(sample, np.mean(sample))
    sample = sample / np.std(sample)
    return sample


def time_shift(sample):
    s_len = sample.shape[1]
    div_loc = np.random.randint(0, s_len)
    sample = np.concatenate((sample[:, div_loc:], sample[:, :div_loc]), axis=1)
    return sample


def pitch_shift():
    pass


def combine_audio(sample_1, sample_2):
    w1 = np.random.random()
    w2 = 1 - w1
    sample = normalize(sample_1) * w1 + normalize(sample_2) * w2
    sample = min_max_scale(sample)
    return sample


def add_noise(sample, noise, d_fac=.4):
    sample = normalize(sample) + normalize(noise) * d_fac * np.random.random()
    sample = min_max_scale(sample)
    return sample


def augmentation(sample):
    pass
