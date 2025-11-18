# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadevokeds

# Evoked-wise function for Delay Time (TAU) computation
from core import delay_time

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flat_evokeds, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 14
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'zbmasking'

# Cluster label
clust_lb = 'CFPO'

# Calcultation parameters label
calc_lb = 'MI'

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
sv_path = obs_path(exp_name = exp_name, obs_name = 'delay', avg_trials = avg_trials, clust_lb = clust_lb, calc_lb = calc_lb)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[1:3]
#ch_list = ch_list[0:2]

# Only averaged conditions
conditions = conditions[0:2]

# Compare Frontal and Parieto-occipital clusters
ch_list = ['Fp1'],['Fp2'],['Fpz'],['Fp1', 'Fp2', 'Fpz'],['O2'],['PO4'],['PO8'],['O2', 'PO4', 'PO8'],['Fp1', 'Fp2', 'Fpz','O2', 'PO4', 'PO8']

# Crazy stupid all electrodes average
#ch_list =  ch_list,
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Method for delay estimation
method = 'mutual_information'
#method = 'autocorrelation'

# Method for handling delay time of clusters
clst_method = 'avg'
#clst_method = 'max'

# Window of interest
frc = [0., 1.]

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False
    clst_method = None

# Dictionary for computation variables
variables = {   
                'method': method,
                'window' : frc,
                'clustered' : clt,
                'clst_method': clst_method,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
            }

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadevokeds(subID: str):

    evokeds = loadevokeds(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return evokeds

# Build Correlation Sum iterable function
def it_delay_time(evoked: mne.Evoked):

    tau = delay_time(evoked = evoked, ch_list = ch_list,
                          method = method, clst_method = clst_method,
                          fraction = frc)

    return tau

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
def mp_delay_time(evoks_iters: list, points: list):

    print('\nComputing Delay Time over each trial')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results = list(tqdm(p.imap(it_delay_time, evoks_iters, chunksize = chunksize),
                            desc = 'Computing channels time series',
                            unit = 'trl',
                            total = len(evoks_iters),
                            leave = True,
                            dynamic_ncols = True)
                        )

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list)]

    tau = collapse_trials(results = results, points = points, fshape = fshape)
    
    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + 'delay.npz', *tau)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults common shape: ', tau[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', tau[c].shape[0])

    print('')

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    DELAY TIME SCRIPT')

    evoks_iters, points = mp_loadevokeds()

    mp_delay_time(evoks_iters = evoks_iters, points = points)

