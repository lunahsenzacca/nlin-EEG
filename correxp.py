# Usual suspects
import os
import json
import warnings

import numpy as np

from tqdm import tqdm

# Scipy function for linear regression
from scipy.stats import linregress

# Utility function for corrsum.py results loading
from core import correxp_getcorrsum

# Utility function for observables directories and data
from core import obs_path, obs_data

# Sub-wise function for correlation exponent
from core import correlation_exponent

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 10
chunksize = 1

### LOAD PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Get data averaged across trials
avg_trials = False

# Label for load results files
clust_lb = 'CFPO'

# Label for saved results files
sv_lb = '3noavg'

# Correlation Sum results directory
path = obs_path(exp_name = exp_name, obs_name = 'corrsum', clust_lb = clust_lb, avg_trials = avg_trials)

# Correlation Exponent saved results directory
sv_path = obs_path(exp_name = exp_name, obs_name = 'correxp', clust_lb = clust_lb, calc_lb = sv_lb, avg_trials = avg_trials)

### SCRIPT PARAMETERS ###

# Average correlation sum over electrodes
avg = False

# Number of points in moving average
n_points = 3

# Get log_r for initzialization
with open(path + 'variables.json', 'r') as f:
    variables = json.load(f)

log_r = variables['log_r']

### SCRIPT FOR COMPUTATION ###

# Build iterable function
def it_correlation_exponent(sub_log_CS: np.ndarray):

    CE, E_CE = correlation_exponent(sub_log_CS = sub_log_CS, avg_trials = avg_trials, n_points = n_points, log_r = log_r)

    return CE, E_CE

# Build multiprocessing function
def mp_correlation_exponent():

    print('\nPreparing Correlation Sum results for computation')

    # Build iterable over subject
    log_CS_iters, variables = correxp_getcorrsum(path = path, avg = avg)

    # Check if mobile average leaves more than three cooridnates
    rlen = len(log_r) - n_points + 1
    if rlen < 4:
        print('\n\'n_points\' too big, choose a smaller value')
        return

    # Get reduced array for mobile average
    r_log_r = []
    for i in range(0,rlen):
        r_log_r.append(np.mean(np.asarray(log_r)[i:i + n_points]))

    print('\nComputing Correlation Exponent over each subject')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results = list(tqdm(p.imap(it_correlation_exponent, log_CS_iters), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'sub',
                       total = len(log_CS_iters),
                       leave = True,
                       dynamic_ncols = True)
                        )

    CE = []
    E_CE = []

    for r in results:
        
        CE.append(r[0])
        E_CE.append(r[1])
    
    CE = np.asarray(CE)
    E_CE = np.asarray(E_CE)

    # Save value and error as one array
    CE = np.concatenate((CE[np.newaxis], E_CE[np.newaxis]), axis = 0)

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'correxp.npy', CE)

    variables['log_r'] = r_log_r
    variables['avg'] = avg
    variables['n_points'] = n_points

    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f)

    print('\nResults shape: ', CE.shape, '\n')

    return

# Launch script with 'python -m idim' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION EXPONENT SCRIPT')

    mp_correlation_exponent()