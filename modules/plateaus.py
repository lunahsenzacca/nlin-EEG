# Usual suspects
import os
import json
import warnings

import numpy as np

from tqdm import tqdm

# Utility function for correxp.py results loading
from core import pp_getcorrexp

# Utility function for observables directories and data
from core import obs_path, obs_data

# Sub-wise function for plateau detection
from core import ce_plateaus

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

# Labels for load results files
clust_lb = 'CFPO'

# Calculation parameters label for load results
calc_lb = '[[m_dense_MI]w_None]3nogauss'

# Correlation Exponent saved results directory
path = obs_path(exp_name = exp_name, obs_name = 'correxp', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'plateaus', clust_lb = clust_lb, calc_lb = calc_lb, avg_trials = avg_trials)

### SCRIPT PARAMETERS ###

# Number of points to consider for noise region detection
screen_points = 70

# Resolution used for noise scaling breakoff
resolution = 10

# Number of points to subtract from noise region start
backsteps = 5

# Maximum number of points in the plateau
max_points = 50

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
def it_ce_plateaus(trial_CE: np.ndarray):

    P, Pe, Pr = ce_plateaus(trial_CE = trial_CE, log_r = log_r,
                            screen_points = screen_points,
                            resolution = resolution,
                            backsteps = backsteps,
                            max_points = max_points)

    return P, Pe, Pr

# Build multiprocessing function
def mp_ce_plateaus():

    print('\nFinding Correlation Exponent Plateaus')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Build iterable over subject
    CE_iters, points, variables = pp_getcorrexp(path = path)

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_ce_plateaus, CE_iters), #chunksize = chunksize),
                       desc = 'Computing trials ',
                       unit = 'trl',
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
    Pr = collapse_trials(results = pr_results, points = points, fshape = [*fshape,2], e_results = None)

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'plateaus.npz', *P)
    np.savez(sv_path + 'plateaus_r.npz', *Pr)

    variables['screen_points'] = screen_points
    variables['resolution'] = resolution
    variables['backsteps'] = backsteps
    variables['max_points'] = max_points

    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f, indent = 3)

    print('\nResults common shape: ', P[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', P[c].shape[0])

    print('')

    return

# Launch script with 'python -m idim' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION EXPONENT PLATEAUS SCRIPT')

    mp_ce_plateaus()