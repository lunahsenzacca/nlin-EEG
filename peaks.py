# Usual suspects
import os
import json
import warnings

import numpy as np

from tqdm import tqdm

# Utility function for observables directories and data
from core import obs_path, obs_data

# Sub-wise function for peaks and plateau detection
from core import ce_peaks

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 10
chunksize = 1

### LOAD PARAMETERS ###

# Dataset name
exp_name = 'zbmasking'

# Get data averaged across trials
avg_trials = True

# Labels for load results files
clust_lb = 'mCFPOdense'
calc_lb = '5noavg'

# Correlation Exponent saved results directory
path = obs_path(exp_name = exp_name, obs_name = 'correxp', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'peaks', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

# Get log_r for initzialization
with open(path + 'variables.json', 'r') as f:
    variables = json.load(f)

log_r = variables['log_r']

### SCRIPT FOR COMPUTATION ###

# Build iterable function
def it_ce_peaks(sub_CE: np.ndarray):

    __, __, P, Pr = ce_peaks(sub_CE = sub_CE, log_r = log_r)

    return P, Pr

# Build multiprocessing function
def mp_ce_peaks():

    print('\nFinding Correlation Exponent Peak')
    print('\nSpawning ' + str(workers) + ' processes...')

    CE, E_CE, __, __ = obs_data(obs_path = path, obs_name = 'correxp', compound_error = not(avg_trials))

    CE = np.concatenate((CE[np.newaxis], E_CE[np.newaxis]), axis = 0)

    sub_CE_iters = [CE[:,i] for i in range(0,CE.shape[1])]

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results = list(tqdm(p.imap(it_ce_peaks, sub_CE_iters), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'sub',
                       total = len(sub_CE_iters),
                       leave = True,
                       dynamic_ncols = True)
                        )

    #D2 = []
    #D2r = []

    P = []
    Pr = []

    for r in results:
        
        #D2.append(r[0])
        #D2r.append(r[1])
        P.append(r[0])
        Pr.append(r[1])
    
    #D2 = np.asarray(D2)
    #D2r = np.asarray(D2r)
    P = np.asarray(P)
    Pr = np.asarray(Pr)

    P = np.swapaxes(P, 0, 1)

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'peaks.npy', P)
    np.save(sv_path + 'peaks_r.npy', Pr)

    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f)

    print('\nResults shape: ', P.shape, '\n')

    return

# Launch script with 'python -m idim' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION EXPONENT PEAKS SCRIPT')

    mp_ce_peaks()