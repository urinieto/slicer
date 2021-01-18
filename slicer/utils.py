import librosa
import numpy as np

import colorednoise as cn

from .constants import N_MELS
from .constants import SR
from .constants import N_FFT
from .constants import HOP_LENGTH
from .constants import MEL_FMIN
from .constants import MEL_FMAX
from .constants import PINK_NOISE_DUR



def compute_melspec(audio):
    """Computes a mel-spectrogram from the given audio data."""
    return librosa.feature.melspectrogram(y=audio,
                                          sr=SR,
                                          n_mels=N_MELS,
                                          n_fft=N_FFT,
                                          hop_length=HOP_LENGTH,
                                          fmin=MEL_FMIN,
                                          fmax=MEL_FMAX)


def pad_pink_noise(mel):
    """Pads pink noise at beginning and end of mel-spectrogram"""
    pink_noise_len = int(PINK_NOISE_DUR * SR)
    start = compute_melspec(cn.powerlaw_psd_gaussian(1, size=pink_noise_len))
    end = compute_melspec(cn.powerlaw_psd_gaussian(1, size=pink_noise_len))
    return np.hstack((start, mel, end))


def to_db(mel):
    """Converts melspectrogram to dB"""
    return librosa.power_to_db(mel, ref=np.max)
