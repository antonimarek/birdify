import numpy as np
from pydub import AudioSegment


def audio_to_array(in_path: str, in_format: str, sr: bool = False, ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    """
    Converts audio signal from file to numpy array.

    :param sr: If True, returns sampling rate
    :param in_path: Path of audio file
    :param in_format: Format of audio file
    :param ffmpeg_path: Path of ffmpeg codec (important when decoding mp3)
    :return: Audio signal as numpy.array, if sr=True returns tuple(signal, sr)
    """
    if in_format == 'mp3':
        AudioSegment.converter = ffmpeg_path
    sample = AudioSegment.from_file(in_path, in_format)
    srt = sample.frame_rate
    sample = sample.get_array_of_samples()
    sample = np.array(sample, dtype=float)
    if sr:
        return sample, srt
    else:
        return sample
