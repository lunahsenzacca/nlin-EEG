# Usual suspects
import os
import json

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadMNE

# Trial-wise function for persistance diagrams
from core import persistence

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flatMNEs, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'persistence'

### MULTIPROCESSING PARAMETERS ###

workers = 10
chunksize = 1

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

### PARAMETERS FOR FREQUENCY SPECTRUM COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

### ADD PARAMETERS

max_pairs = parameters['max_pairs']

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clst = True
else:
    clst = False

# Dictionary for computation variables
variables = {   
                'window' : window,
                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'max_pairs': max_pairs,
### ADD PARAMETERS
            }

### DATA PATHS ###

if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

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

# Build persistance diagrams iterable function 
def it_persistence(MNE_l: list):

    PS, TPS = persistence(MNE = MNE_l[0], ch_list = ch_list, max_pairs = max_pairs, window = window)

    return PS, TPS

# Build evoked/epochs loading multiprocessing function
def mp_loadMNE():

    print('\nLoading data')#\n\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        loaded = list(tqdm(p.imap(it_loadMNE, sub_list),#, chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False,
                       dynamic_ncols = True)
                       )
    
    # Create flat iterable list of MNE objects
    MNEs_iters, points = flatMNEs(MNEs = loaded)

    print('\nDONE!')

    return MNEs_iters, points

# Build persistence multiprocessing function
def mp_persistence(MNEs_iters: list, points: list):

    print('\nComputing Fourier Transform over each trial')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_persistence, MNEs_iters, chunksize = chunksize),
                            desc = 'Computing channels time series',
                            unit = 'trl',
                            total = len(MNEs_iters),
                            leave = True,
                            dynamic_ncols = True)
                        )
    resultsPS = []
    resultsTPS = []
    for r in results_:

        resultsPS.append(r[0])
        resultsTPS.append(r[1])
    
    # Create homogeneous arrays for results
    fshape = [len(sub_list),len(conditions),len(ch_list),max_pairs,4]

    PS = collapse_trials(results = resultsPS, points = points, fshape = fshape, dtype = np.float64)

    fshape = [len(sub_list),len(conditions),len(ch_list),max_pairs,2]

    TPS = collapse_trials(results = resultsTPS, points = points, fshape = fshape, dtype = np.int32)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *PS)
    np.savez(sv_path + f'{obs_name}_times.npz', *TPS)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables, f, indent = 2)

    print('\nResults common shape: ', PS[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', PS[c].shape[0])

    print('')

    return

# Script main method
if __name__ == '__main__':

    print('\n    PERSISTENCE DIAGRAM SCRIPT')

    MNEs_iters, points = mp_loadMNE()

    mp_persistence(MNEs_iters = MNEs_iters, points = points)

