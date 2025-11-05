# Usual suspects
import os
import json
import mne
import warnings

import numpy as np

from tqdm import tqdm

# Cython file compile wrapper
from core import cython_compile

# Sub-wise function for evoked file loading
from core import loadevokeds

# Evoked-wise function for Correlation Sum (CS) computation
from core import correlation_sum

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flat_evokeds, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 10
chunksize = 1

### SCRIPT PARAMETERS ###

# Cython implementation of the script
cython = True
cython_verbose = False

# Dataset name
exp_name = 'zbmasking_dense'

# Cluster label
clust_lb = 'CFPO'

# Calcultation parameters label
calc_lb = '[m_dense_MI]w_80'

# Get data averaged across trials
avg_trials = True

if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### LOAD DATASET DIRECTORIES AND INFOS ###

# Evoked folder paths
path = maind[exp_name]['directories'][method]

# List of ALL subject IDs
sub_list = maind[exp_name]['subIDs']

# List of ALL conditions
conditions = list(maind[exp_name]['conditions'].values())

# List of ALL electrodes
ch_list = maind[exp_name]['pois']

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'corrsum', avg_trials = avg_trials, clust_lb = clust_lb, calc_lb = calc_lb)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[1:4]
#ch_list = ch_list[0:2]

# Only averaged conditions
conditions = conditions[0:2]

# Compare Frontal and Parieto-occipital clusters
ch_list = ['Fp1', 'Fp2', 'Fpz','PO3', 'PO4', 'Oz']#['Fp1'],['Fp2'],['Fpz'],['PO3'],['PO4'],['Oz'],['Fp1', 'Fp2', 'Fpz'],['PO3', 'PO4', 'Oz'],['Fp1', 'Fp2', 'Fpz','PO3', 'PO4', 'Oz']

# Crazy stupid all electrodes average
#ch_list =  ch_list,
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Embedding dimensions
embeddings = [3, 6, 9, 12, 15]

# Set different time delay for each time series
tau = 'mutual_information'
# Or set a global value
#tau = maind[exp_name]['tau']

# Theiler window
w = 80

# Window of interest
frc = [0., 1.]

# Distances for sampling the dependance
log_span = [-2.5, 1, 150, 10]

r = np.logspace(log_span[0], log_span[1], num = log_span[2], base = log_span[3])

# Apply embedding normalization when computing distances
m_norm = True

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False

# Dictionary for computation variables
variables = {   
                'tau' : tau,
                'w': w,
                'window' : frc,
                'm_norm': m_norm,
                'clustered' : clt,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'embeddings' : embeddings,
                'log_span': log_span,
                'log_r': list(np.log(r))
            }

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadevokeds(subID: str):

    evokeds = loadevokeds(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return evokeds

# Build Correlation Sum iterable function
def it_correlation_sum(evoked: mne.Evoked):

    CS = correlation_sum(evoked = evoked, ch_list = ch_list,
                         embeddings = embeddings, tau = tau, w = w, fraction = frc,
                         rvals = r, m_norm = m_norm, cython = cython)

    return CS

# Build evoked loading multiprocessing function
def mp_loadevokeds():

    print('\nPreparing evoked data')#\n\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        evokeds = list(tqdm(p.imap(it_loadevokeds, sub_list),#, chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False,
                       dynamic_ncols = True)
                       )

    # Create flat iterable list of evokeds images
    evoks_iters, points = flat_evokeds(evokeds = evokeds)

    print('\nDONE!')

    return evoks_iters, points

# Build Correlation Sum multiprocessing function
def mp_correlation_sum(evoks_iters: list, points: list):

    # Get absolute complexity of the script and estimated completion time
    complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)*np.log(len(r))*(((maind[exp_name]['T'])**2)*(frc[1]-frc[0])**2)

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
        
        results = list(tqdm(p.imap(it_correlation_sum, evoks_iters, chunksize = chunksize),
                            desc = 'Computing channels time series',
                            unit = 'trl',
                            total = len(evoks_iters),
                            leave = True,
                            dynamic_ncols = True)
                        )

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r)]

    CS = collapse_trials(results = results, points = points, fshape = fshape, dtype = np.float64)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'corrsum.npz', *CS)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults common shape: ', CS[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', CS[c].shape[0])

    print('')

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION SUM PLOT SCRIPT')

    if cython == True:

        cython_compile(verbose = cython_verbose)

    evoks_iters, points = mp_loadevokeds()

    mp_correlation_sum(evoks_iters = evoks_iters, points = points)

