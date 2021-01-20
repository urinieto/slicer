# Paths
ANNS_DIR = "harmonixset/dataset/segments"
MELS_DIR = "melspecs"
DATA_DIR = "data"

# Audio parameters
SR = 22050
HOP_LENGTH = 1024
N_MELS = 80
N_FFT = 2048
MEL_FMIN = 0
MEL_FMAX = None

# Pre-processing parameters
CONTEXT_DUR = 15        # 15-second patches
SMEARING_DUR = 2        # 2-second smearing of the boundary
GAUSSIAN_STD = 5        # Standard deviation of the boundary smearing sampling
PATCH_PER_BOUND = 2     # Number of positive patches per boundary
GAUSSIAN_LEN = int(SMEARING_DUR / (HOP_LENGTH / SR))
PINK_NOISE_DUR = (CONTEXT_DUR + SMEARING_DUR) / 2
PATCH_LEN = int(CONTEXT_DUR / (HOP_LENGTH / SR))

# Peak picking parameters
PEAK_PICKING_CONTEXT = 6
PEAK_PICKING_THRES = 0.5
PEAK_PICKING_PAST_AVG = 12
PEAK_PICKING_FUTU_AVG = 6

# Architecture parameters
N_CONV_LAYERS = 5
N_DENSE_LAYERS = 2
N_MINIBATCH = 64
