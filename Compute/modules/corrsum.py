# Usual suspects
import os
import json

import numpy as np

from tqdm import tqdm

# Cython file compile wrapper
from core import cython_compile

# Sub-wise function for MNE file loading
from core import loadMNE

# Evoked-wise function for Correlation Sum (CS) computation
from core import correlation_sum

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flatMNEs, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'corrsum'

### MULTIPROCESSING PARAMETERS ###

workers = 10
chunksize = 1

### CYTHON DEBUG PARAMETERS ###

# Cython implementation of the script
cython = False
cython_verbose = False

### LOAD EXPERIMENT INFO AND SCRIPT PARAMETERS ###

with open('./.tmp/info.json', 'r') as f:

    info = json.load(f)

with open(f'./.tmp/modules/{obs_name}.json', 'r') as f:

    parameters = json.load(f)

### EXPERIMENT PARAMETERS ###

# Dataset name
exp_name = info['exp_name']

# Cluster label
clst_lb = info['clst_lb']

# Averaging method
avg_trials = info['avg_trials']

# Subject IDs
sub_list = info['sub_list']

# Conditions
conditions = info['conditions']

# Channels
ch_list = info['ch_list']

# Time window
window = info['window']

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clst = True
else:
    clst = False

# Get method string
if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Embedding dimensions
embeddings = parameters['embeddings']

# Set different time delay for each time series
tau = parameters['tau']

# Theiler window
w = parameters['w']

# Apply embedding normalization when computing distances
m_norm = parameters['m_norm']

# Distances for sampling the dependance
log_span = parameters['log_span']

r = np.logspace(log_span[0], log_span[1], num = log_span[2], base = log_span[3])

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'tau' : tau,
                'w': w,
                'embeddings' : embeddings,
                'm_norm': m_norm,
                'log_span': log_span,
                'log_r': list(np.log(r)),

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

### DATA PATHS ###

# Processed data
path = maind[exp_name]['directories'][method]

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = obs_name, avg_trials = avg_trials, clst_lb = clst_lb, calc_lb = calc_lb)

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadMNE(subID: str):

    MNEs = loadMNE(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return MNEs

# Build Correlation Sum iterable function
def it_correlation_sum(MNE_l: list):

    CS = correlation_sum(MNE = MNE_l[0], ch_list = ch_list,
                         embeddings = embeddings, tau = tau, w = w, window = window,
                         rvals = r, m_norm = m_norm, cython = cython)

    return CS

# Build evoked loading multiprocessing function
def mp_loadMNEs():

    print('\nPreparing evoked data')#\n\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        MNEs = list(tqdm(p.imap(it_loadMNE, sub_list),#, chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False,
                       dynamic_ncols = True)
                       )

    # Create flat iterable list of evokeds images
    MNEs_iters, points = flatMNEs(MNEs = MNEs)

    print('\nDONE!')

    return MNEs_iters, points

# Build Correlation Sum multiprocessing function
def mp_correlation_sum(MNEs_iters: list, points: list):

    if window is None:
        i_window = maind[exp_name]['window']
    else:
        i_window = window

    # Get absolute complexity of the script and estimated completion time
    complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)*np.log(len(r))*(((maind[exp_name]['T'])**2)*(i_window[1]-i_window[0])**2)

    velocity = 26e-7

    import datetime
    eta = str(datetime.timedelta(seconds = int(complexity*velocity/workers)))

    print('\nComputing Correlation Sum over each trial')
    print('\nNumber of single computations: ' + str(int(complexity)))
    print('\nEstimated completion time < ~' + eta)
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results = list(tqdm(p.imap(it_correlation_sum, MNEs_iters, chunksize = chunksize),
                            desc = 'Computing channels time series',
                            unit = 'trl',
                            total = len(MNEs_iters),
                            leave = True,
                            dynamic_ncols = True))

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r)]

    CS = collapse_trials(results = results, points = points, fshape = fshape, dtype = np.float64)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *CS)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables, f, indent = 2)

    with open(sv_path + 'info.json','w') as f:
        json.dump(info, f, indent = 2)

    print('\nResults common shape: ', CS[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')

        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', CS[c].shape[0])

    print('')

    return

# Script main method
if __name__ == '__main__':

    print('\n    CORRELATION SUM PLOT SCRIPT')

    if cython == True:

        cython_compile(verbose = cython_verbose)

    MNEs_iters, points = mp_loadMNEs()

    mp_correlation_sum(MNEs_iters = MNEs_iters, points = points)

