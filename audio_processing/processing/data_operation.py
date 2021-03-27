from pydub import AudioSegment
import librosa
import sox

# def fs_from_mp3(path):
#     signal = AudioSegment.from_mp3(path)
#     fs = signal.frame_rate
#     signal, _ = librosa.load(path, sr=fs)
#     data.at[f, 'Time'] = signal.shape[0]/fs

sample_rate = sox.file_info.sample_rate('path/to/file.mp3')