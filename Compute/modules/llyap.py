# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadMNE

# Sub-wise function for Largest Lyapunov Exponent (LLE) computation
from core import lyapunov

# Utility function for observables directories
from core import obs_path

# Utility functions for trial data averaging
from core import flatMNEs, collapse_trials

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'llyap'

### MULTIPROCESSIN PARAMETERS ###
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

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False

# Get method string
if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### PARAMETERS FOR LARGEST LYAPUNOV EXPONENT COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Embedding dimensions
embeddings = parameters['embeddings']

# Set different time delay for each time series
tau = parameters['tau']

# Theiler window
w = parameters['w']

# Lenght of expansion phase
dt = parameters['dt']

# Apply embedding normalization when computing distances
m_norm = parameters['m_norm']

# Dictionary for computation variables
variables = {   
                'tau' : tau,
                'w': w,
                'window' : window,
                'clustered' : clt,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'embeddings' : embeddings
            }

### DATA PATHS ###

# Processed data
path = maind[exp_name]['directories'][method]

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = obs_name, avg_trials = avg_trials, clst_lb = clst_lb, calc_lb = calc_lb)

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadMNE(subID: str):

    evokeds = loadMNE(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return evokeds

# Build Largest Lyapunov Exponent iterable function
def it_lyapunov(MNE_l: list):

    LY, E_LY = lyapunov(MNE = MNE_l[0], ch_list = ch_list,
                        embeddings = embeddings, tau = tau,
                        dt = dt, w = w, window = window, verbose = False)

    return [LY, E_LY]

# Build evoked loading multiprocessing function
def mp_loadMNEs():

    print('\nPreparing evoked data')#\n\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        MNEs = list(tqdm(p.imap(it_loadMNE, sub_list), #chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False)
                       )

    # Create flat iterable list of evokeds images
    MNEs_iters, points = flatMNEs(MNEs = MNEs)

    print('\nDONE!')

    return MNEs_iters, points

# Build multiprocessing function
def mp_lyapunov(MNEs_iters: list, points: list):

    complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)

    velocity = 0.7

    import datetime
    eta = str(datetime.timedelta(seconds = int(complexity*velocity/workers)))

    print('\nComputing Largest Lyapunov Exponent over each trial')
    print('\nNumber of single computations: ' + str(complexity))
    print('\nEstimated completion time < ~' + eta)
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results = list(tqdm(p.imap(it_lyapunov, MNEs_iters), #chunksize = chunksize),
                                        desc = 'Computing channels time series',
                                        unit = 'trl',
                                        total = len(MNEs_iters),
                                        leave = True,
                                        dynamic_ncols = True)
                                    )

    # Get separate results lists
    r = [[ly for ly in trial[0]] for trial in results]
    e_r = [[e_ly for e_ly in trial[1]] for trial in results]

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings)]

    LY = collapse_trials(results = r, points = points, fshape = fshape, e_results = e_r)

    print('\nDONE!')

   # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *LY)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults common shape: ', LY[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', LY[c].shape[0])

    print('')

    return

# Launch script with 'python -m llyap' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    LARGEST LYAPUNOV EXPONENT SCRIPT')

    MNEs_iters, points = mp_loadMNEs()

    mp_lyapunov(MNEs_iters = MNEs_iters, points = points)

