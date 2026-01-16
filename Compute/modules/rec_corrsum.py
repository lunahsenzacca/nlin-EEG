# Usual suspects
import os
import json
import warnings

import numpy as np

from tqdm import tqdm

# Utility function for reurrence.py results loading
from core import corrsum_getrecurrence

# Utility function for observables directories and data
from core import obs_path, obs_data

# Trial-wise function for Correlation Sum computation
from core import correlation_sum

# Function for results formatting
from core import collapse_trials

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 14
chunksize = 1

### LOAD PARAMETERS ###

# Dataset name
exp_name = 'zbmasking_dense'

# Get data averaged across trials
avg_trials = True

# Cluster label
clust_lb = 'F'

# Calculation parameters label for load results
calc_lb = 'test'

# Calculation parameters label for saved results
sv_calc_lb = 'w_0'

# Make explicit reference to previus calculation parameters
sv_calc_lb = '[' + calc_lb + ']' + sv_calc_lb

# Correlation Exponent saved results directory
path = obs_path(exp_name = exp_name, obs_name = 'recurrence', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'corrsum', clust_lb = clust_lb, calc_lb = sv_calc_lb, avg_trials = avg_trials)

### SCRIPT PARAMETERS ###

# Theiler window
w = None

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
def it_correlation_sum(trial_RP: np.ndarray):

    CS, E_CS = correlation_sum(trial_RP = trial_RP, log_r = log_r, w = w)

    return CS, E_CS

# Build multiprocessing function
def mp_correlation_sum():

    print('\nComputing Correlation Sum from Recurrence Plots')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Build iterable over subject
    RP_iters, points, variables = corrsum_getrecurrence(path = path)

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_correlation_sum, RP_iters, chunksize = chunksize),
                       desc = 'Computing trials ',
                       unit = 'trl',
                       total = len(RP_iters),
                       leave = True,
                       dynamic_ncols = True)
                        )

    cs_results = []
    e_cs_results = []
    for r in results_:

        cs_results.append(r[0])
        e_cs_results.append(r[1])

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(log_r)]

    CS = collapse_trials(results = cs_results, points = points, fshape = fshape, e_results = e_cs_results)

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'corrsum.npz', *CS)

    variables['Theiler window'] = w
    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f, indent = 2)

    print('\nResults common shape: ', CS[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', CS[c].shape[0])

    print('')

    return

# Script main method
if __name__ == '__main__':

    print('\n    CORRELATION SUM SCRIPT')

    mp_correlation_sum()
