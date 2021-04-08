from pydub import AudioSegment
import os
from tqdm import tqdm


def resample_to_file(in_path: str, out_path: str, in_format: str = 'mp3', out_format: str = 'mp3', sr: int = 22050,
                     channels: int = 1, ffmpeg_path: str = r"C:/ffmpeg/bin/ffmpeg.exe"):
    AudioSegment.converter = ffmpeg_path
    sample = AudioSegment.from_file(in_path, in_format)
    sample = sample.set_frame_rate(sr)
    if channels:
        sample = sample.set_channels(channels)
    sample.export(out_path, out_format)
    pass


def resample_dir(dir_path_in: str, dir_path_out: str, in_format: str = 'mp3', out_format: str = 'mp3', sr: int = 22050,
                 channels: int = 1):
    f_list = [f for f in os.listdir(dir_path_in) if '.' in f]
    for _, file in enumerate(tqdm(f_list, desc="Progress")):
        in_path = dir_path_in + '/' + file
        out_path = dir_path_out + '/' + file
        resample_to_file(in_path=in_path, out_path=out_path, in_format=in_format, out_format=out_format, sr=sr,
                         channels=channels)
    return f'{len(f_list)} files from {dir_path_in} resampled into {dir_path_out}'
