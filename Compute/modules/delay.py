# Usual suspects
import os
import json

import numpy as np

# Multiprocessing wrapper
from core import mp_wrapper

# Sub-wise function for MNE file loading
from core import loadMNE

# Evoked-wise function for Delay Time (TAU) computation
from core import delay_time

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flatMNEs, collapse_trials

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

obs_name = 'delay'

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

### PARAMETERS FOR DELAY TIME COMPUTATION ###

# Choose delay time computation method
tau_method = parameters['tau_method']

# Choose clustering method
clst_method = parameters['clst_method']

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

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'tau_method': tau_method,
                'clst_method': clst_method,

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
                   avg_trials = avg_trials, conditions = conditions,
                   with_std = False)

    return MNEs

# Build Correlation Sum iterable function
def it_delay_time(MNE: list):

    tau = delay_time(MNE = MNE[0], ch_list = ch_list,
                          tau_method = tau_method, clst_method = clst_method,
                          window = window)

    return tau 

# Build MNE loading multiprocessing function
def mp_loadMNE():

    #print('\nLoading data')#\n\nSpawning ' + str(workers) + ' processes...')

    MNEs = mp_wrapper(it_loadMNE, iterable = sub_list,
                      workers = workers,
                      chunksize = chunksize,
                      desc = 'Loading data',
                      unit = 'sub')

    # Create flat iterable list of MNE objects
    MNEs_iters, points = flatMNEs(MNEs = MNEs)

    print('\nDONE!')

    return MNEs_iters, points

# Build Correlation Sum multiprocessing function
def mp_delay_time(MNEs_iters: list, points: list):

    print('\nComputing Delay Time over each trial')
    print('\nSpawning ' + str(workers) + ' processes...')

    results_ = mp_wrapper(it_delay_time, iterable = MNEs_iters,
                          workers = workers,
                          chunksize = chunksize,
                          desc = 'Computing',
                          unit = 'trl')

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list)]

    tau = collapse_trials(results = results_, points = points, fshape = fshape)
    
    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *tau)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables, f, indent = 2)

    with open(sv_path + 'info.json','w') as f:
        json.dump(info, f, indent = 2)

    print('\nResults common shape: ', tau[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', tau[c].shape[0])

    print('')

    return

# Script main method
if __name__ == '__main__':

    print('\n    DELAY TIME SCRIPT')

    MNEs_iters, points = mp_loadMNE()

    mp_delay_time(MNEs_iters = MNEs_iters, points = points)

