# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

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
workers = 20
chunksize = 10

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Label for results folder
lb = 'CFPOr'

# Get data averaged across trials
avg_trials = False

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
sv_path = obs_path(exp_name = exp_name, obs_name = 'corrsum', clust_lb = lb, avg_trials = avg_trials)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[0:2]
#ch_list = ch_list[0:2]

#Only averaged conditions
conditions = conditions[0:2]
#Parieto-Occipital and Frontal electrodes
ch_list = ['Fp1','Fp2','Fpz'],['O2','PO4','PO8']
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Embedding dimensions
embeddings = [i for i in range(2,21)]

# Time delay
tau = maind[exp_name]['tau']

# Window of interest
frc = [0, 1]

# Distances for sampling the dependance
r = np.logspace(1.7, 2.38, num = 8, base = 10)
r = r/1e7

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False

# Dictionary for computation variables
variables = {   
                'tau' : tau,
                'window' : frc,
                'clustered' : clt,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'embeddings' : embeddings
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
                         embeddings = embeddings, tau = tau, fraction = frc,
                         rvals = r)

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
    complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)*len(r)

    velocity = 0.52

    import datetime
    eta = str(datetime.timedelta(seconds = int(complexity*velocity/workers)))

    print('\nComputing Correlation Sum over each trial')
    print('\nNumber of single computations: ' + str(complexity))
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

    CS = collapse_trials(results = results, points = points, fshape = fshape)
    
    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'corrsum.npy', CS)

    variables['shape'] = CS.shape
    variables['log_r'] = list(np.log(r))

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults shape: ', CS.shape, '\n')

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION SUM SCRIPT')

    evoks_iters, points = mp_loadevokeds()

    mp_correlation_sum(evoks_iters = evoks_iters, points = points)

