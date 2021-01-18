import argparse
import glob
import os
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from slicer import utils
from slicer.constants import GAUSSIAN_LEN
from slicer.constants import GAUSSIAN_STD
from slicer.constants import ANNS_DIR
from slicer.constants import MELS_DIR
from slicer.constants import SR
from slicer.constants import HOP_LENGTH
from slicer.constants import CONTEXT_DUR
from slicer.constants import PINK_NOISE_DUR
from slicer.constants import SMEARING_DUR
from slicer.constants import PATCH_PER_BOUND
from slicer.constants import DATA_DIR


class Patcher(object):
    def __init__(self):
        # Get frame indices from time durations
        self.offset = librosa.time_to_frames(
            PINK_NOISE_DUR, sr=SR, hop_length=HOP_LENGTH)
        self.context_f = librosa.time_to_frames(
            CONTEXT_DUR, sr=SR, hop_length=HOP_LENGTH)
        self.smearing_f = librosa.time_to_frames(
            SMEARING_DUR, sr=SR, hop_length=HOP_LENGTH)

        # Compute gaussian to be used during the whole process
        self.gaussian = compute_gaussian()

    def sample_and_weight(self):
        """Samples from the Gaussian and includes weight (i.e., likelihood)"""
        p = self.gaussian / np.sum(self.gaussian)
        samp = np.random.choice(np.arange(len(self.gaussian)), p=p)
        weight = self.gaussian[samp] / np.max(self.gaussian)
        return samp - len(self.gaussian) // 2, weight

    def get_patch(self, X, mid):
        """Gets a patch given the features and mid point"""
        start = mid - self.context_f // 2
        end = mid + self.context_f // 2
        return X[:, start:end]

    def get_positive_patch(self, X, bound_f):
        """Gets a positive boundary patch, given a features and boundary"""
        samp, weight = self.sample_and_weight()
        bound_f += samp + self.offset
        return self.get_patch(X, bound_f), weight

    def get_negative_patch(self, X, bounds_f):
        """Gets a negative boundary patch."""
        # Create uniform probabilities removing bound areas with context
        N = X.shape[1] - self.offset * 2
        p = np.ones(N)
        for b in bounds_f:
            start = max(0, b - self.smearing_f // 2)
            end = min(b + self.smearing_f // 2, N)
            p[start:end] = 0

        # Sample from probabilities
        p = p / np.sum(p)
        bound_f = np.random.choice(np.arange(N), p=p) + self.offset

        # Return patch and return
        return self.get_patch(X, bound_f), 1.0


def compute_gaussian():
    """Computes a gaussian window with default parameters"""
    return signal.windows.gaussian(GAUSSIAN_LEN, GAUSSIAN_STD)


def load_bounds(ann_file):
    """Reads the boundaries from the given annotation file"""
    with open(ann_file, "r") as f:
        d = f.readlines()
    return np.array([float(row.split(" ")[0]) for row in d])


def load_mel(mel_file):
    """Reads the mel-spectrogram, add pink noise, and converts to db"""
    return utils.to_db(utils.pad_pink_noise(np.load(mel_file)))


def process_track(mel_file, ann_file):
    """Process a given track with the mel-spectrogram and annotation files."""
    # print(mel_file)
    # Read annotations
    bounds = load_bounds(ann_file)

    # Load mel spectrogram, pad pink noise, and convert to dB
    X = load_mel(mel_file)

    patches = []
    weights = []
    labels = []
    bounds_f = librosa.time_to_frames(bounds, sr=SR, hop_length=HOP_LENGTH)
    patcher = Patcher()
    for bound_f in bounds_f:
        for _ in range(PATCH_PER_BOUND):
            pos_patch, pos_w = patcher.get_positive_patch(X, bound_f)
            neg_patch, neg_w = patcher.get_negative_patch(X, bounds_f)

            # Store
            patches += [pos_patch, neg_patch]
            weights += [pos_w, neg_w]
            labels += [1, 0]
    return patches, weights, labels


def process_split(mel_files, anns_dir, out_dir, suffix):
    """Processes a data split with the given mel files."""
    X = []
    W = []
    Y = []
    for mel_file in tqdm(mel_files):
        name = os.path.basename(mel_file)
        ann_file = os.path.join(anns_dir, name.replace("-mel.npy", ".txt"))
        x, w, y = process_track(mel_file, ann_file)
        X += x
        W += w
        Y += y

    # Make sure we have the same data sizes
    assert(len(X) == len(W) and len(W) == len(Y))

    # Suffle data
    indices = np.random.choice(np.arange(len(X)), len(X), replace=False)
    X = np.asarray(X)[indices]
    W = np.asarray(W)[indices]
    Y = np.asarray(Y)[indices]

    # Save
    np.save(os.path.join(out_dir, f"X_{suffix}.npy"), X)
    np.save(os.path.join(out_dir, f"W_{suffix}.npy"), W)
    np.save(os.path.join(out_dir, f"Y_{suffix}.npy"), Y)


def process_all_tracks(mels_dir, anns_dir, data_dir):
    # Read input files and make sure file lengths match
    mel_files = glob.glob(os.path.join(mels_dir, "*.npy"))
    ann_files = glob.glob(os.path.join(anns_dir, "*.txt"))
    assert(len(mel_files) == len(ann_files))

    # Split data
    mel_files_train, mel_files_test = \
        train_test_split(mel_files, test_size=0.2, random_state=1)

    # Process for each split
    process_split(mel_files_train, anns_dir, data_dir, "train")
    process_split(mel_files_test, anns_dir, data_dir, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mels_dir',
                        dest='mels_dir',
                        default=MELS_DIR,
                        help='Directory with mel-spectrograms')
    parser.add_argument('--anns_dir',
                        dest='anns_dir',
                        default=ANNS_DIR,
                        help='Directory with segmentation annotations')
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        default=DATA_DIR,
                        help='Directory to store the dataset')

    args = parser.parse_args()
    np.random.seed(666)
    process_all_tracks(**vars(args))
