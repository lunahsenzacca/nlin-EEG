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

# Function for results formatting
from core import collapse_trials

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 8
chunksize = 1

### LOAD PARAMETERS ###

# Dataset name
exp_name = 'zbmasking_dense'

# Get data averaged across trials
avg_trials = True

# Cluster label
clust_lb = 'CFPO'

# Calculation parameters label for load results
calc_lb = '[m_dense_MI]w_80'

# Calculation parameters label for saved results
sv_calc_lb = '3gauss'

# Make explicit reference to previus calculation parameters
sv_calc_lb = '[' + calc_lb + ']' + sv_calc_lb

# Correlation Sum results directory
path = obs_path(exp_name = exp_name, obs_name = 'corrsum', avg_trials = avg_trials, clust_lb = clust_lb, calc_lb = calc_lb)

# Correlation Exponent saved results directory
sv_path = obs_path(exp_name = exp_name, obs_name = 'correxp', avg_trials = avg_trials, clust_lb = clust_lb, calc_lb = sv_calc_lb)

### SCRIPT PARAMETERS ###

# Number of points in moving average for derivative computation
n_points = 3

# Apply gaussian filter to results for smoothing
gauss_filter = True

# Parameters of the gaussian filter
scale = 0.01
cutoff = 7

if gauss_filter == False:
    scale = None
    cutoff = None

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
def it_correlation_exponent(sub_log_CS: np.ndarray):

    CE, E_CE = correlation_exponent(sub_log_CS = sub_log_CS,
                                    n_points = n_points, gauss_filter = gauss_filter,
                                    scale = scale, cutoff = cutoff, log_r = log_r)

    return CE, E_CE

# Build multiprocessing function
def mp_correlation_exponent():

    print('\nPreparing Correlation Sum results for computation')

    # Build iterable over subject
    log_CS_iters, points, variables = correxp_getcorrsum(path = path)

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
        
        results_ = list(tqdm(p.imap(it_correlation_exponent, log_CS_iters), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'sub',
                       total = len(log_CS_iters),
                       leave = True,
                       dynamic_ncols = True)
                        )

    results = []
    e_results = []
    for r in results_:

        results.append(r[0])
        e_results.append(r[1])

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r_log_r)]

    CE = collapse_trials(results = results, points = points, fshape = fshape, e_results = e_results)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'correxp.npz', *CE)

    variables['log_r'] = r_log_r
    variables['n_points'] = n_points

    variables['gauss'] = gauss_filter

    if gauss_filter == True:
        variables['scale'] = scale
        variables['cutoff'] = cutoff

    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f, indent = 3)

    print('\nResults common shape: ', CE[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', CE[c].shape[0])

    print('')

    return

# Launch script with 'python -m idim' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION EXPONENT SCRIPT')

    mp_correlation_exponent()