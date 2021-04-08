from typing import List

import numpy as np
# import pandas as pd
from audio_processing.processing.extraction.feature_preparation import min_max_scale
from tensorflow.keras.utils import Sequence, to_categorical


# from tensorflow.data import Dataset


class AudioDataGenerator(Sequence):
    def __init__(self, pandas_df, noise_IDs: List[str], sample_dir: str, noise_dir: str, batch_size=32, dim=(54, 256),
                 n_channels=1, n_classes=None, over: bool = True, augment: bool = True, shuffle=True):
        """Initialization"""
        self.over = over
        self.augment = augment
        self.dim = dim
        self.sample_dir = sample_dir
        self.batch_size = batch_size
        self.pandas_df = pandas_df
        self.noise_IDs = noise_IDs
        self.noise_dir = noise_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        self.max_samples_class = self.pandas_df.catg.value_counts()[0]
        return int(np.floor(self.max_samples_class * self.n_classes / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.temp_df.iloc[k].ID for k in indexes]
        list_labels = [self.temp_df.iloc[k].catg for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        samples_class = self.pandas_df.catg.value_counts()
        self.temp_df = self.pandas_df
        if self.over:
            for curr_class in samples_class.index[1:]:
                add_samples = samples_class.values[0] - samples_class[samples_class.index == curr_class].values[0]
                if add_samples > 0:
                    gen_sample = self.pandas_df[self.pandas_df.catg == curr_class].sample(add_samples, replace=True)
                    self.temp_df.append(gen_sample)

        self.indexes = np.arange(len(self.temp_df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_labels):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sample = np.load(self.sample_dir + ID)
            if self.augment:
                sample = time_shift(sample)
                noise = np.load(self.noise_dir + np.random.choice(self.noise_IDs))
                sample = add_noise(sample, noise)

            sample = min_max_scale(sample)
            shp = sample.shape
            X[i] = sample.reshape((*shp, 1))

            # Store class
            y[i] = list_labels[i]

        return X, to_categorical(y, num_classes=self.n_classes)


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
