import librosa
import numpy as np


def pre_process_file_mfcc(file_path):
    audio, sampling_rate = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(y=audio,
                                sr=sampling_rate,
                                n_mfcc=40,
                                n_mels=80,
                                n_fft=551,
                                hop_length=220)

    mfcc = (mfcc - mfcc.mean(axis=-1)[:, np.newaxis]) / mfcc.std(axis=-1)[:, np.newaxis]
    return mfcc


def pre_process_file_mel_log(file_path):
    audio, sampling_rate = librosa.load(file_path)

    mel = librosa.feature.melspectrogram(y=audio,
                                         sr=sampling_rate,
                                         n_mels=80,
                                         n_fft=551,
                                         hop_length=220,
                                         fmax=11025)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db
