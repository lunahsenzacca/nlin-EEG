import json
import numpy as np

from scipy.spatial.distance import squareform

from core import obs_path

def rrate(RP: np.ndarray, exclude_trivial: bool = True, raw_RP: bool = False) -> float:

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

    num = np.sum(RP)
    den = len(RP)**2

    if exclude_trivial is True:

        num -= len(RP)

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
def it_rrate(parameters: dict):

    global iterable

    def iterable(RP: np.ndarray):

        RR = rrate(RP = RP,
                          exclude_trivial = parameters['exclude_trivial'],
                          raw_RP = True)

        return [RR]

    return iterable
