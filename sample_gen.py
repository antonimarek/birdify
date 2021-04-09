# import numpy as np
import pandas as pd
# from soundmatrix.processing.extraction import resample_dir, split_signal_noise_dir, feature_extraction
from soundmatrix.processing.extraction import serial_chunks

'''
Resampling and signal/noise separation
'''
# local_sample_dir = r'./samples/signal/'
# local_noise_dir = r'./samples/noise/'

# resample_dir('./mp3', './resampled', sr=22050, channels=1)

# split_signal_noise_dir('./resampled', './resampled')
'''
Chunk generation
'''
data = pd.read_csv('dataset.csv')

files = data.Path.to_list()
class_names = data.Name.apply(lambda x: x.replace(' ', '_'))
classes = data.class_label.map(str)
class_ids = classes.str.cat(class_names, sep="_").to_list()

# generate signal chunks
in_signal_path = r'./samples/signal/'
out_signal_path = r'./samples/chunks_signal/'
in_signal_list = [in_signal_path+f for f in files]
serial_chunks(in_paths=in_signal_list, in_format='mp3', class_ids=class_ids, length=3, out_dir=out_signal_path,
              out_format='npz')

# generate noise chunks
# in_noise_path = r'./samples/noise/'
# out_noise_path = r'./samples/chunks_noise/'


# files = data.Path.to_list()
# classes = data.class_label.to_list()
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
# feature_extraction(files, classes, local_sample_dir, leftover=True, padding=True, deltas=False,
#                    feat_type='mfcc', n_mel=128, s_len=130, to_dir='./numpy_mfcc/', full_length=False)
#
# # extract noise samples randomly
# np.random.shuffle(files)
# feature_extraction(files[:200], classes, local_noise_dir, leftover=True, padding=True, deltas=False,
#                    feat_type='mfcc', s_len=130, n_mel=128, to_dir='./numpy_mfcc/noise/', full_length=False)


