import mir_eval
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from . import utils
from .constants import DATA_DIR
from .constants import ANNS_DIR
from .constants import MELS_DIR
from .constants import TEST_NAMES


def eval_slicer(S):
    """Evals the given slicer with the Harmonix Set."""
    # Open test track names
    with open(TEST_NAMES, "r") as f:
        test_names = f.readlines()

    res = []
    for name in tqdm(test_names):
        # Read and process annotations
        name = name.replace("\n", "")
        ref_bounds = utils.load_bounds(os.path.join(ANNS_DIR, name + ".txt"))
        ref_inters = utils.to_intervals(ref_bounds)

        # Compute and process estimations
        mel = np.load(os.path.join(MELS_DIR, name + "-mel.npy"))
        est_bounds = S.do_the_slice(mel)
        est_bounds = np.concatenate(([0], est_bounds, [mel.shape[1] - 1]))
        est_inters = utils.to_intervals(utils.to_times(est_bounds))

        # Compute and store evaluation metrics
        res.append(mir_eval.segment.evaluate(
            ref_inters,
            [0] * len(ref_inters),
            est_inters,
            [0] * len(est_inters)))

    return pd.DataFrame(res)
