import numpy as np
import pandas as pd
from audio_processing.processing import resample_dir, split_signal_noise_dir, feature_extraction

# resample_dir('./mp3', './resampled', sr=22050, channels=1)

# split_signal_noise_dir('./resampled', './resampled')

data = pd.read_csv('dataset.csv')

files = data.Path.to_list()
classes = data.class_label.to_list()

'''
Mel Spectrogram features extraction
'''
# # extract feature samples
# feature_extraction(files, classes, r'./samples/signal/', leftover=True, padding=True,
#                    feat_type='mel_spec', n_mel=64, to_dir='./numpy_mfcc/', full_length=False, s_len=256)
#
# # extract noise samples randomly
# np.random.shuffle(files)
# feature_extraction(files[:100], classes, r'./samples/noise/', leftover=True, feat_type='mel_spec',
#                    n_mel=64, to_dir='./numpy_mfcc/noise/', full_length=False)

'''
MFCC features extraction
time length ~= 3s
'''
# extract feature samples
feature_extraction(files, classes, r'./samples/signal/', leftover=True, padding=True,
                   feat_type='mfcc', n_mel=128, to_dir='./numpy_mfcc/', full_length=False, s_len=130)

# extract noise samples randomly
np.random.shuffle(files)
feature_extraction(files[:200], classes, r'./samples/noise/', leftover=True, feat_type='mfcc', s_len=130,
                   n_mel=128, to_dir='./numpy_mfcc/noise/', full_length=False)
