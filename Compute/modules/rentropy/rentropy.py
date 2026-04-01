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

def rentropy(RP: np.ndarray, min_lengths: list = [3], exclude_trivial: bool = True, raw_RP: bool = False) -> np.ndarray:
    """
    RS: Shannon entropy of the diagonal recurrence lines probability distibution

    Parameters
    ----------
    RP : (n,n) array-like
        Binary recurrence plot.
    min_lengths : list
        Minimum diagonal line lengths (inclusive).
    exclude_trivial : bool
        If True, excludes the line of identity (offset=0), as is common in RQA.

    Returns
    -------
    rs : float
        RQA entropy based on probability distribution of diagonal recurrence lines.
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

    lengths = []
    for offset in range(-(n - 1), 0):
        if exclude_trivial and offset == 0:
            continue
        rl = _run_lengths_ones(np.diagonal(RP, offset=offset))
        if rl.size == 0:
            continue

        [lengths.append(l) for l in rl]

    N = len(lengths)

    unique, counts = np.unique(np.asarray(lengths), return_counts = True)

    dist = dict(zip(unique,counts/N))

    terms = []
    for l in dist.keys():

        terms.append(-dist[l]*np.log(dist[l]))

    terms = np.asarray(terms)

    rs = []

    for l in min_lengths:

        rs.append(terms[l:].sum())

    return np.asarray(rs)

# Prepare recurrence results
def get_recurrence(info: dict, load_calc_lb: str):

    path = obs_path(exp_name = info['exp_name'], obs_name = 'recurrence', avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = load_calc_lb)

    # Load Recurence Plot file
    RP = np.load(path + 'recurrence.npz')

    return RP

# Iterable function generator
def it_rentropy(parameters: dict):

    global iterable

    def iterable(RP: np.ndarray):

        RS = rentropy(RP = RP,
                      min_lengths = parameters['min_dlengths'],
                      exclude_trivial = parameters['exclude_trivial'],
                      raw_RP = True)

        return [RS]

    return iterable
