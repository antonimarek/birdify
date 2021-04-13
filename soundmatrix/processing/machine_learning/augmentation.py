import numpy as np
# from soundmatrix.processing.extraction.feature_preparation import min_max_scale
import librosa
# from numba import jit


# @jit(nopython=True)
def normalize(sample):
    sample = np.subtract(sample, np.mean(sample))
    sample = sample / np.std(sample)
    return sample


def time_shift(sample):
    s_len = sample.shape[-1]
    div_loc = np.random.randint(0, s_len)
    if len(sample.shape) == 1:
        return np.concatenate((sample[div_loc:], sample[:div_loc]))
    elif len(sample.shape) == 2:
        return np.concatenate((sample[:, div_loc:], sample[:, :div_loc]), axis=1)
    else:
        return ValueError("Sample can't have more than 2 dimensions!")


def pitch_shift(sample, sr: int = 22050, half_tone_var: float = 2):
    sample = librosa.effects.pitch_shift(y=sample, sr=sr,
                                         n_steps=np.random.uniform(-half_tone_var, half_tone_var),
                                         bins_per_octave=24)
    return sample


# @jit(nopython=True)
def combine_audio(sample_1, sample_2):
    w1 = np.random.uniform(.5, 1)
    w2 = 1 - w1
    sample = normalize(sample_1) * w1 + normalize(sample_2) * w2
    sample = normalize(sample)
    return sample


# @jit(nopython=True)
def add_noise(sample, noise, d_fac=.4, r_mltp: bool = True):
    if r_mltp:
        r_multiplier = np.random.random()
    else:
        r_multiplier = 1
    sample = normalize(sample) + normalize(noise) * d_fac * r_multiplier
    sample = normalize(sample)
    return sample


def augment_audio(sample, noise_sample=None, comb_sample=None, sr: int = 22050):
    sample = time_shift(sample)
    # sample = pitch_shift(sample, sr=sr)
    if comb_sample is not None:
        # comb_sample = time_shift(comb_sample)
        # comb_sample = pitch_shift(comb_sample, sr=sr)
        sample = combine_audio(sample, comb_sample)
    if noise_sample is not None:
        # noise_sample = time_shift(noise_sample)
        # noise_sample = pitch_shift(noise_sample, sr=sr)
        sample = add_noise(sample, noise_sample, r_mltp=False)
    sample = normalize(sample)
    return sample
