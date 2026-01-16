# Usual suspects
import os
import json

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadMNE

# Evoked-wise function for Evokeds Plotting (EV)
from core import evokeds

# Utility function for observables directories
from core import obs_path

# Utility function for dimensional time and frequency domain of the experiment
from core import get_tinfo

# Utility functions for trial data management
from core import flatMNEs, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

obs_name = 'evokeds'

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

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clst = True
else:
    clst = False

if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

# Load frequency domain informations and get freqencies array
_, times = get_tinfo(exp_name = exp_name, avg_trials = avg_trials, window = window)

# Dictionary for computation variables
variables = {   
                'window' : window,
                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                't': list(times)
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
                          avg_trials = avg_trials, conditions = conditions, with_std = True)

    return MNEs

# Build Evoked Plotting iterable function
def it_evokeds(MNE_l: list):

    EP, E_EP = evokeds(MNE = MNE_l[0], s_MNE = MNE_l[1], ch_list = ch_list, window = window)

    return EP, E_EP

# Build evoked loading multiprocessing function
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

    print(len(MNEs_iters),len(MNEs_iters[0]))

    print('\nDONE!')

    return MNEs_iters, points

# Build Evokeds Plotting multiprocessing function
def mp_evokeds(MNEs_iters: list, points: list):

    print('\nExtracting Evoked Signal')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_evokeds, MNEs_iters, chunksize = chunksize),
                            desc = 'Computing channels time series',
                            unit = 'trl',
                            total = len(MNEs_iters),
                            leave = True,
                            dynamic_ncols = True))

    # Get separate results lists
    results = []
    e_results = []
    for r in results_:

        results.append(r[0])
        e_results.append(r[1])

    lenght = len(times)

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),lenght]

    EV = collapse_trials(results = results, points = points, fshape = fshape, dtype = np.float64, e_results = e_results)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *EV)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables, f, indent = 2)

    with open(sv_path + 'info.json','w') as f:
        json.dump(info, f, indent = 2)

    print('\nResults common shape: ', EV[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', EV[c].shape[0])

    print('')

    return

# Script main method
if __name__ == '__main__':

    print('\n    EVOKEDS PLOT SCRIPT')

    MNEs_iters, points = mp_loadMNE()

    mp_evokeds(MNEs_iters = MNEs_iters, points = points)

