# Usual suspects
import os
import json
import warnings

import numpy as np

from tqdm import tqdm

# Utility function for correxp.py results loading
from core import peaks_getcorrexp

# Utility function for observables directories and data
from core import obs_path, obs_data

# Sub-wise function for peaks and plateau detection
from core import ce_peaks

# Function for results formatting
from core import collapse_trials

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 10
chunksize = 1

### LOAD PARAMETERS ###

# Dataset name
exp_name = 'zbmasking_dense'

# Get data averaged across trials
avg_trials = True

# Labels for load results files
clust_lb = 'mCFPOVANdense'
calc_lb = '3gauss'

# Correlation Exponent saved results directory
path = obs_path(exp_name = exp_name, obs_name = 'correxp', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'peaks', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

### SCRIPT PARAMETERS ###

# Distance of peak from other
distance = None

# Height boundaries of peaks
height = (0.8,6)

# Prominence boundaries of peaks
prominence = (0.3,None)

# Width boundaries of peaks
width = (3,30)

# Get log_r for initzialization
with open(path + 'variables.json', 'r') as f:
    variables = json.load(f)

sub_list = variables['subjects']
conditions = variables['conditions']
ch_list = variables['pois']
embeddings = variables['embeddings']
log_r = variables['log_r']

### SCRIPT FOR COMPUTATION ###

# Build iterable function
def it_ce_peaks(trial_CE: np.ndarray):

    P, Pe, Pr = ce_peaks(trial_CE = trial_CE, log_r = log_r,
                         distance = distance, height = height,
                         prominence = prominence, width = width)

    return P, Pe, Pr

# Build multiprocessing function
def mp_ce_peaks():

    print('\nFinding Correlation Exponent Peak')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Build iterable over subject
    CE_iters, points, variables = peaks_getcorrexp(path = path)

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_ce_peaks, CE_iters), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'sub',
                       total = len(CE_iters),
                       leave = True,
                       dynamic_ncols = True)
                        )

    p_results = []
    e_p_results = []
    pr_results = []
    for r in results_:

        p_results.append(r[0])
        e_p_results.append(r[1])
        pr_results.append(r[2])

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings)]

    P = collapse_trials(results = p_results, points = points, fshape = fshape, e_results = e_p_results)
    Pr = collapse_trials(results = pr_results, points = points, fshape = fshape, e_results = None)

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'peaks.npz', *P)
    np.savez(sv_path + 'peaks_r.npz', *Pr)

    variables['distance'] = distance
    variables['height'] = height
    variables['prominence'] = prominence
    variables['width'] = width

    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f)

    print('\nResults common shape: ', P[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', P[c].shape[0])

    print('')

    return

# Launch script with 'python -m idim' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION EXPONENT PEAKS SCRIPT')

    mp_ce_peaks()