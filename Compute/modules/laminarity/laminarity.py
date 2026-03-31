import json
import numpy as np

from scipy.spatial.distance import squareform

from core import obs_path

# Utility function for counting lengths of ones
def _run_lengths_ones(x: np.ndarray) -> np.ndarray:
    """Lengths of consecutive 1-runs in a 1D array."""
    x01 = (x != 0).astype(np.int8)           # signed: important for diff
    d = np.diff(np.r_[0, x01, 0])
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    return ends - starts

def laminarity(RP: np.ndarray, min_lengths: list = [3], raw_RP: bool = False) -> np.ndarray:
    """
    LAM: fraction of recurrent points that are part of vertical lines
    of length >= min_length.

    Parameters
    ----------
    RP : (n,n) array-like
        Binary recurrence plot.
    min_lengths : list
        Minimum vertical line lengths (inclusive).

    Returns
    -------
    det : float
        laminarity in [0,1]. Returns 0.0 if no vertical recurrence points exist.
    """

    if raw_RP is True:

        RP = squareform(RP)

        c = len(RP)
        for i in range(len(RP)):
            RP[i,i] = 0
            if RP[i,0] == 2:
                c = i
                break

        RP = RP[:c,:c]

    if RP.ndim != 2:
        raise ValueError("RP must be a 2D array.")
    n, m = RP.shape
    if n != m:
        raise ValueError("RP must be square.")

    num = np.asarray([0 for i in range(0,len(min_lengths))])  # points in columns with length >= min_length
    den = 0  # points in all columns (length >= 1)

    for j in range(0, n):
        rl = _run_lengths_ones(RP[j:,j])
        if rl.size == 0:
            continue
        den += rl.sum()

        for i, l in enumerate(min_lengths):
            num[i] += rl[rl >= l].sum()

    if den == 0:
        return np.asarray([0.0 for i in range(0,len(min_lengths))])

    return num / den

# Prepare recurrence results
def get_results(info: dict, load_calc_lb: str):

    path = obs_path(exp_name = info['exp_name'], obs_name = 'recurrence', avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = load_calc_lb)

    # Load Recurence plot file
    RP = np.load(path + 'recurrence.npz')

    # Load calculation info
    with open(path + 'info.json', 'r') as f:
        info = json.load(f)

    return RP, info

# Iterable function generator
def it_laminarity(parameters: dict):

    global iterable

    def iterable(RP: np.ndarray):

        LAM = laminarity(RP = RP,
                          min_lengths = parameters['min_vlengths'],
                          raw_RP = True)

        return [LAM]

    return iterable
