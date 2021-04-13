import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from ..extraction import get_feature
from .augmentation import augment_audio
from typing import List
import tensorflow as tf


class AudioDataGenerator:
    """
    Tensorflow (Keras) generator for audio data.
    Supported f_type: {'mpcc','mpcc_deltas','mel_spec'}
    """

    def __init__(self, pandas_df, noise_IDs: List[str], sample_dir: str, noise_dir: str = None, batch_size=32,
                 n_channels=1, n_classes=None, over: bool = True, augment: bool = True, shuffle=True,
                 feature_params: dict = {'f_type': 'mel_spec'}):

        self.feature_params = feature_params
        self.over = over
        self.augment = augment
        self.sample_dir = sample_dir
        self.batch_size = batch_size
        self.pandas_df = pandas_df
        self.noise_IDs = noise_IDs
        self.noise_dir = noise_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.shape = self.shape()
        self.over_sample()

    def shape(self):
        test_sample = self.pandas_df.iloc[0].ID
        sample = np.load(self.sample_dir + test_sample)['a']
        sample = get_feature(sample, **self.feature_params)
        shape = sample.shape
        return shape, sample

    def over_sample(self):
        if self.over:
            samples_class = self.pandas_df.catg.value_counts()
            temp_df = self.pandas_df
            if self.over:
                for curr_class in samples_class.index[1:]:
                    add_samples = samples_class.values[0] - samples_class[samples_class.index == curr_class].values[0]
                    if add_samples > 0:
                        gen_sample = self.pandas_df[self.pandas_df.catg == curr_class].sample(add_samples, replace=True)
                        temp_df.append(gen_sample)
            self.pandas_df = temp_df
        pass

    def data_generator(self):
        if self.augment:
            # random assignment of second sample of same class and random noise from whole input set
            self.pandas_df['ID2'] = self.pandas_df['catg'].apply(lambda x:
                                                                     np.random.choice(
                                                                         self.pandas_df[self.pandas_df.catg == x].ID)
                                                                     )
            self.pandas_df['ID_noise'] = \
                pd.Series(self.noise_IDs).sample(len(self.pandas_df), replace=True).values

            # creating tf dataset
            dataset = tf.data.Dataset.from_tensor_slices((self.pandas_df.ID,
                                                          self.pandas_df.ID2,
                                                          self.pandas_df.ID_noise,
                                                          self.pandas_df.catg))
            # shuffle must be placed before mapping functions for speed reasons
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=len(self.pandas_df))
            # mapping functions to perform augmentation on each sample randomly
            dataset = dataset.map(self.augment_samples,
                                  num_parallel_calls=tf.data.AUTOTUNE)  # returns X, y
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.pandas_df.ID,
                                                          self.pandas_df.catg))
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=len(self.pandas_df))
            dataset = dataset.map(self.load_signal,
                                  num_parallel_calls=tf.data.AUTOTUNE)

        # mapping function to perform feature extraction from signals
        dataset = dataset.map(self.generate_features,
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        return dataset

    def load_signal(self, X, y):
        X = tf.numpy_function(
            func=self.load_single, inp=[X], Tout=tf.float32)
        return X, y

    def load_single(self, X):
        X = X.decode()
        X = np.load(self.sample_dir + X)['a']
        return X

    def augment_samples(self, X, sample2, noise, y):
        # Loads and augments signal
        X = tf.numpy_function(
            func=self.random_augmentation, inp=[X, sample2, noise], Tout=tf.double)
        return X, y

    def random_augmentation(self, X, sample2, noise):
        X = X.decode()
        sample2 = sample2.decode()
        noise = noise.decode()
        X = np.load(self.sample_dir + X)['a']
        # performs augmentation
        aug_params = {}
        if self.noise_dir is not None:
            noise = np.load(self.noise_dir + noise)['a']
            aug_params['noise_sample'] = noise
        sample2 = np.load(self.sample_dir + sample2)['a']
        aug_params['comb_sample'] = sample2
        X = augment_audio(X, **aug_params)
        return X

    def generate_features(self, X, y):
        X = tf.numpy_function(
            func=self.x_feature, inp=[X], Tout=tf.float32)
        y = tf.numpy_function(
            func=to_categorical, inp=[y, self.n_classes], Tout=tf.float32)
        return X, y

    def x_feature(self, X):
        X = get_feature(X, **self.feature_params)
        return X
