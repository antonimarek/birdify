import numpy as np
from tensorflow.keras.utils import to_categorical
from ..extraction.feature_preparation import min_max_scale, get_feature
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
        self.shape = self.shape()

    def shape(self):
        test_sample = self.pandas_df.iloc[0].ID
        sample = np.load(self.sample_dir + test_sample)['a']
        sample = get_feature(sample, **self.feature_params)
        shape = (*sample.shape, self.n_channels)
        return shape

    def data_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.pandas_df.ID)
        dataset_cat = tf.data.Dataset.from_tensor_slices(self.pandas_df.catg)

        dataset_cat = dataset_cat.map(lambda item: tf.numpy_function(
                                      self.to_cat, [item], np.float),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda item: tf.numpy_function(
                              self.map_func, [item], np.double),
                              num_parallel_calls=tf.data.AUTOTUNE
                              )

        dataset = tf.data.Dataset.zip((dataset, dataset_cat))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.batch_size*10)

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        return dataset

    def map_func(self, X):
        X = X.decode('UTF-8')
        y = int(X[0])

        f_p = self.sample_dir + X
        X = np.load(f_p)['a']
        if self.augment:
            aug_params = {}
            if self.noise_dir is not None:
                noise = np.load(self.noise_dir + np.random.choice(self.noise_IDs))['a']
                aug_params['noise_sample'] = noise
            sample2_id = np.random.choice(a=self.pandas_df[self.pandas_df.catg == y].ID.values, size=1)
            sample2 = np.load(self.sample_dir + sample2_id[0])['a']
            aug_params['comb_sample'] = sample2
            X = augment_audio(X, **aug_params)
        X = get_feature(X, **self.feature_params)
        X = min_max_scale(X)
        X = X.reshape(self.shape)
        return X

    def to_cat(self, y):
        y = to_categorical(y, num_classes=self.n_classes)
        return y
