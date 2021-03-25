from sklearn.base import BaseEstimator, TransformerMixin


class SoundPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.name = None
