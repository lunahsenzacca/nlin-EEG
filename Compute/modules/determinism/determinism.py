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

def determinism(RP: np.ndarray, min_length: int = 3, exclude_trivial: bool = True, raw_RP: bool = False) -> float:
    """
    DET: fraction of recurrent points that are part of diagonal lines
    of length >= min_length.

    Parameters
    ----------
    RP : (n,n) array-like
        Binary recurrence plot.
    min_length : int
        Minimum diagonal line length (inclusive).
    exclude_trivial : bool
        If True, excludes the line of identity (offset=0), as is common in RQA.

    Returns
    -------
    det : float
        Determinism in [0,1]. Returns 0.0 if no diagonal recurrence points exist.
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

    num = 0  # points in diagonals with length >= min_length
    den = 0  # points in all diagonals (length >= 1)

    for offset in range(-(n - 1), 0):
        if exclude_trivial and offset == 0:
            continue
        rl = _run_lengths_ones(np.diagonal(RP, offset=offset))
        if rl.size == 0:
            continue
        den += rl.sum()
        num += rl[rl >= min_length].sum()

    if den == 0:
        return 0.0

    return num / den

# Prepare recurrence results
def get_recurrence(info: dict, load_calc_lb: str):

    path = obs_path(exp_name = info['exp_name'], obs_name = 'recurrence', avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = load_calc_lb)

    # Load Recurence plot file
    RP = np.load(path + 'recurrence.npz')

    # Load calculation info
    with open(path + 'info.json', 'r') as f:
        info = json.load(f)

    return RP

# Iterable function generator
def it_determinism(parameters: dict):

    global iterable

    def iterable(RP: np.ndarray):

        DET = determinism(RP = RP,
                          min_length = parameters['min_dlength'],
                          exclude_trivial = parameters['exclude_trivial'],
                          raw_RP = True)

        return [DET]

    return iterable
