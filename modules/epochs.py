# Usual suspects
import os
import json
import mne
import warnings

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadevokeds

# Evoked-wise function for Epochs Plotting (EP)
from core import epochs

# Utility function for observables directories
from core import obs_path

# Utility function for dimensional time and frequency domain of the experiment
from core import get_tinfo

# Utility functions for trial data averaging
from core import flat_evokeds, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 10
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'zbmasking_dense'

# Cluster label
clust_lb = 'CFPO'

# Calcultation parameters label
calc_lb = 'FL'

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
sv_path = obs_path(exp_name = exp_name, obs_name = 'epochs', avg_trials = avg_trials, clust_lb = clust_lb, calc_lb = calc_lb)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[1:4]
#ch_list = ch_list[0:2]

# Only averaged conditions
#conditions = conditions[:2]

# Compare Frontal and Parieto-occipital clusters
ch_list = ['Fp1', 'Fp2', 'Fpz','PO3', 'PO4', 'Oz']#['Fp1'],['Fp2'],['Fpz'],['PO3'],['PO4'],['Oz'],['Fp1', 'Fp2', 'Fpz'],['PO3', 'PO4', 'Oz'],['Fp1', 'Fp2', 'Fpz','PO3', 'PO4', 'Oz']

# Crazy stupid all electrodes average
#ch_list =  ch_list,
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Window of interest
window = None

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False

# Get time coordinates
_, times = get_tinfo(exp_name = exp_name, method = method, window = window)

# Dictionary for computation variables
variables = {   
                'window' : window,
                'clustered' : clt,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                't': list(times)
            }

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadevokeds(subID: str):

    evokeds = loadevokeds(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return evokeds

# Build evokeds loading iterable function
def it_loadevokeds_std(subID: str):

    s_evokeds = loadevokeds(subID = subID, exp_name = exp_name,
                            avg_trials = avg_trials, conditions = conditions, std = True)

    return s_evokeds

# Build Epochs Plotting iterable function
def it_epochs(evoked_l: list):

    EP, E_EP = epochs(evoked = evoked_l[0], s_evoked = evoked_l[1], ch_list = ch_list, window = window)

    return EP, E_EP

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

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        s_evokeds = list(tqdm(p.imap(it_loadevokeds_std, sub_list),#, chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False,
                       dynamic_ncols = True)
                       )

    # Create flat iterable list of evokeds images
    s_evoks_iters, points = flat_evokeds(evokeds = s_evokeds)

    evoks_iters = [[evoks_iters[i],s_evoks_iters[i]] for i in range(0,len(evoks_iters))]

    print('\nDONE!')

    return evoks_iters, points

# Build Epochs Plotting multiprocessing function
def mp_epochs(evoks_iters: list, points: list):

    print('\nComputing Epoched Time Series over each trial')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_epochs, evoks_iters, chunksize = chunksize),
                            desc = 'Computing channels time series',
                            unit = 'trl',
                            total = len(evoks_iters),
                            leave = True,
                            dynamic_ncols = True)
                        )
    results = []
    e_results = []
    for r in results_:

        results.append(r[0])
        e_results.append(r[1])

    lenght = len(times)

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),lenght]

    EP = collapse_trials(results = results, points = points, fshape = fshape, dtype = np.float64, e_results = e_results)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'epochs.npz', *EP)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults common shape: ', EP[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', EP[c].shape[0])

    print('')

    return

# Launch script with 'python -m recurrence' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    EPOCHS PLOT SCRIPT')

    evoks_iters, points = mp_loadevokeds()

    mp_epochs(evoks_iters = evoks_iters, points = points)

