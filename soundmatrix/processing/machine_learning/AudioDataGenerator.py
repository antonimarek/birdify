import os

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from ..extraction.feature_preparation import min_max_scale, get_feature
from .augmentation import augment_audio
from typing import List
import tensorflow as tf


class AudioDataGenerator(Sequence):
    """
    Tensorflow (Keras) generator for audio data.
    Supported f_type: {'mpcc','mpcc_deltas','mel_spec'}
    """

    def __init__(self, pandas_df, noise_IDs: List[str], sample_dir: str, noise_dir: str = None, batch_size=32,
                 n_channels=1, n_classes=None, over: bool = True, augment: bool = True, shuffle=True,
                 feature_params: dict = {'f_type': 'mpcc'}):

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
        self.on_epoch_end()
        self.shape = self.shape()
        # self.list_noise()

    def __len__(self):

        self.max_samples_class = self.pandas_df.catg.value_counts()[0]
        return int(np.floor(self.max_samples_class * self.n_classes / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.temp_df.iloc[k].ID for k in indexes]
        list_labels = [self.temp_df.iloc[k].catg for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels)

        return X, y

    def shape(self):
        test_sample = self.pandas_df.iloc[0].ID
        sample = np.load(self.sample_dir + test_sample)['a']
        sample = get_feature(sample, **self.feature_params)
        shape = (*sample.shape, self.n_channels)
        return shape

    def on_epoch_end(self):
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

        # Initialization
        # X = np.empty((self.batch_size, *self.shape[:-1], self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        dataset = tf.data.Dataset.from_tensor_slices(list_IDs_temp)
        dataset_cat = tf.data.Dataset.from_tensor_slices(list_labels)

        dataset_cat = dataset_cat.map(lambda item: tf.numpy_function(
                                      self.to_cat, [item], np.float),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda item: tf.numpy_function(
                              self.map_func, [item], np.double),
                              num_parallel_calls=tf.data.AUTOTUNE
                              )
        dataset = np.array(list(dataset.as_numpy_iterator()))
        dataset_cat = np.array(list(dataset_cat.as_numpy_iterator()))

        # dataset = tf.data.Dataset.zip((dataset, dataset_cat))

        # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     with open(self.sample_dir + ID, 'rb') as f:
        #         sample = np.load(f)['a']
        #     if self.augment:
        #         aug_params = {}
        #         if self.noise_dir is not None:
        #             with open(self.noise_dir + np.random.choice(self.noise_IDs), 'rb') as f:
        #                 noise = np.load(f)['a']
        #             aug_params['noise_sample'] = noise
        #         id2 = np.random.choice(self.pandas_df[self.pandas_df.catg == list_labels[i]].ID)
        #         with open(self.sample_dir + id2, 'rb') as f:
        #             sample2 = np.load(f)['a']
        #         # sample2 = np.load(self.sample_dir + id2)['a']
        #         aug_params['comb_sample'] = sample2
        #         sample = augment_audio(sample, **aug_params)
        #     sample = min_max_scale(sample)
        #     sample = get_feature(sample, **self.feature_params)
        #     # feature type
        #     shp = sample.shape
        #     X[i] = sample.reshape((*shp, 1))
        #
        #     # Store class
        #     y[i] = list_labels[i]
        # return X, to_categorical(y, num_classes=self.n_classes)
        return dataset, dataset_cat

    def map_func(self, feature_path):
        feature_path = feature_path.decode('UTF-8')
        f_p = self.sample_dir + feature_path
        feature = np.load(f_p)['a']
        if self.augment:
            aug_params = {}
            if self.noise_dir is not None:
                noise = np.load(self.noise_dir + np.random.choice(self.noise_IDs))['a']
                aug_params['noise_sample'] = noise
            sample2_id = np.random.choice(self.pandas_df[self.pandas_df.catg == int(feature_path[0])].ID.values)
            sample2 = np.load(self.sample_dir + sample2_id)['a']
            aug_params['comb_sample'] = sample2
            feature = augment_audio(feature, **aug_params)
        feature = get_feature(feature)
        feature = min_max_scale(feature)
        feature = feature.reshape(self.shape)
        return feature

    def to_cat(self, y):
        y = to_categorical(y, num_classes=self.n_classes)
        return y



"""
class AudioDataGenerator(Sequence):
    def __init__(self, pandas_df, noise_IDs: List[str], sample_dir: str, noise_dir: str, batch_size=32, dim=(54, 256),
                 n_channels=1, n_classes=None, over: bool = True, augment: bool = True, shuffle=True):
        
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
        
        self.max_samples_class = self.pandas_df.catg.value_counts()[0]
        return int(np.floor(self.max_samples_class * self.n_classes / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.temp_df.iloc[k].ID for k in indexes]
        list_labels = [self.temp_df.iloc[k].catg for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels)

        return X, y

    def on_epoch_end(self):
        
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
"""
