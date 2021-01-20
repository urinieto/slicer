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
from .constants import PEAK_PICKING_CONTEXT
from .constants import PEAK_PICKING_PAST_AVG
from .constants import PEAK_PICKING_FUTU_AVG



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


def get_mel_patches(mel, hop_length, patch_length):
    """Gets a set of patches from the given mel spectrogram.
    Warning: Contiguous in memory, do not modify returned values!
    """
    n_bins, n_frames = mel.shape
    mel = np.asfortranarray(mel)  # Row-wise formatting

    # Compute the number of patches that will fit. The end may get truncated.
    n_patches = 1 + int((n_frames - patch_length) / hop_length)

    return np.lib.stride_tricks.as_strided(
            mel,
            shape=(n_patches, n_bins, patch_length),
            strides=(hop_length * n_bins * mel.itemsize,
                     mel.itemsize,
                     n_bins * mel.itemsize))


def peak_picking(x, th):
    """Peak picking from novelty curve x, following Ulrich 2014."""
    context = int(PEAK_PICKING_CONTEXT / (HOP_LENGTH / SR))
    mid = context // 2
    frames = librosa.util.frame(x, context, 1)

    # Get candidates
    candidates = []
    for i, f in enumerate(frames.T):
        if f[mid] > max(np.concatenate((f[:mid], f[mid + 1:]))):
            candidates.append([f[mid], i + mid])

    # Average candidates
    past = int(PEAK_PICKING_PAST_AVG * SR / HOP_LENGTH)
    futu = int(PEAK_PICKING_FUTU_AVG * SR / HOP_LENGTH)
    for c in candidates:
        start = max(0, c[1] - past)
        end = min(len(x), c[1] + futu)
        c[0] -= np.average(x[start:end])

    # Filter by threshold
    candidates = np.asarray(candidates)
    bounds = candidates[np.where(candidates[:, 0] > th)[0], 1]
    return np.asarray(bounds, dtype=int)


def load_bounds(ann_file):
    """Reads the boundaries from the given annotation file"""
    with open(ann_file, "r") as f:
        d = f.readlines()
    return np.array([float(row.split(" ")[0]) for row in d])
