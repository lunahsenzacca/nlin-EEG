# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadevokeds

# Sub-wise function for Largest Lyapunov Exponent (LLE) computation
from core import lyapunov

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flat_evokeds, collapse_trials

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###
workers = 16
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Label for results folder
lb = 'CFPO'

# Get data averaged across trials
avg_trials = False

if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### LOAD DATASET DIRECTORIES AND INFOS ###

# Evoked folder path
path = maind[exp_name]['directories'][method]

# List of ALL subject IDs
sub_list = maind[exp_name]['subIDs']

# List of ALL conditions
conditions = list(maind[exp_name]['conditions'].values())

# List of ALL electrodes
ch_list = maind[exp_name]['pois']

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'llyap', clust_lb = lb, avg_trials = avg_trials)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[0:1]
#ch_list = ch_list[0:3]

#Only averaged conditions
conditions = list(conditions)[0:2]
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

### SCRIPT FOR COMPUTATION ###

# Build evokeds loading iterable function
def it_loadevokeds(subID: str):

    evokeds = loadevokeds(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return evokeds

# Build Largest Lyapunov Exponent iterable function
def it_lyapunov(evoked: mne.Evoked):

    ly, ly_e = lyapunov(evoked = evoked, ch_list = ch_list,
                        embeddings = embeddings, tau = tau,
                        fraction = frc)

    return [ly, ly_e]

# Build evoked loading multiprocessing function
def mp_loadevokeds():

    print('\nPreparing evoked data')#\n\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        evokeds = list(tqdm(p.imap(it_loadevokeds, sub_list), #chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False)
                       )

    # Create flat iterable list of evokeds images
    evoks_iters, points = flat_evokeds(evokeds = evokeds)

    print('\nDONE!')

    return evoks_iters, points

# Build multiprocessing function
def mp_lyapunov(evoks_iters: list, points: list):

    complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)

    velocity = 0.8

    import datetime
    eta = str(datetime.timedelta(seconds = int(complexity*velocity/workers)))

    print('\nComputing largest lyapunov exponent over each trial')
    print('\nNumber of single computations: ' + str(complexity))
    print('\nEstimated completion time: ~' + eta)
    print('\nSpawning ' + str(workers) + ' processes...')

    if avg_trials == True:
        unit = 'sub'
    else:
        unit = 'trl'

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results = list(tqdm(p.imap(it_lyapunov, evoks_iters), #chunksize = chunksize),
                                        desc = 'Computing channels time series',
                                        unit = unit,
                                        total = len(evoks_iters),
                                        leave = True)
                                    )
    
    # Get separate results lists
    r = [[ly for ly in trial[0]] for trial in results]
    e_r = [[e_ly for e_ly in trial[1]] for trial in results]

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings)]

    ly = collapse_trials(results = r, points = points, fshape = fshape, e_results = e_r)
    
    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'llyap.npy', ly)

    variables['shape'] = ly.shape

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults shape: ', ly.shape, '\n')

    return

# Launch script with 'python -m llyap' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    LARGEST LYAPUNOV EXPONENT SCRIPT')

    evoks_iters, points = mp_loadevokeds()

    mp_lyapunov(evoks_iters = evoks_iters, points = points)

