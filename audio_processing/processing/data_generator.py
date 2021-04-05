# from sklearn.base import BaseEstimator, TransformerMixin
#
#
# class SoundPreprocessing(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.name = None

import numpy as np
from tensorflow.keras.utils import Sequence


class AudioDataGenerator(Sequence):
    def __init__(self, X, y, noise, batch_size=32, shuffle=False):
        """Initialization"""
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.indexes = np.arange(len(self.X))
        self.noise = noise
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)