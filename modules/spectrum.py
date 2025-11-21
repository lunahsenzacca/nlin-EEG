# Usual suspects
import os
import json
import mne
import warnings

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadMNE

# Evoked-wise function for Spectrum Plotting (SP)
from core import spectrum

# Utility function for observables directories
from core import obs_path

# Utility function for dimensional time and frequency domain of the experiment
from core import get_tinfo

# Utility functions for trial data averaging
from core import flatMNEs, collapse_trials

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'spectrum'

### MULTIPROCESSING PARAMETERS ###

workers = 10
chunksize = 1

### LOAD EXPERIMENT INFO AND SCRIPT PARAMETERS ###

with open('./.tmp/last.json', 'r') as f:

    info = json.load(f)

with open(f'./.tmp/modules/{obs_name}.json', 'r') as f:

    parameters = json.load(f)

### EXPERIMENT PARAMETERS ###

# Dataset name
exp_name = info['exp_name']

# Cluster label
clust_lb = info['clst_lb']

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

# Label for 
calc_lb = parameters['calc_lb']

# Number of signals to generate for frequency domain error estimation
N = parameters['N']

# Window factor for frequency space resolution
wf = parameters['wf']

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False

# Load frequency domain informations and get freqencies array
info, times = get_tinfo(exp_name = exp_name, avg_trials = avg_trials, window = window)

# Make a fake fft to get the same frequency binning
f_ts = np.zeros(len(times))

_, freqs = mne.time_frequency.psd_array_welch(f_ts,
                    sfreq = info['sfreq'],  # Sampling frequency from the evoked data
                    fmin = info['highpass'], fmax = info['lowpass'],  # Focus on the filter range
                    n_fft = int(len(f_ts)*wf),  # Length of FFT (controls frequency resolution)
                    n_per_seg = int(len(f_ts)/wf),
                    verbose = False)

# Dictionary for computation variables
variables = {   
                'window' : window,
                'clustered' : clt,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'freqs': list(freqs),
                'N': N,
                'window_factor': wf,
            }

### DATA PATHS ###

if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

# Processed data
path = maind[exp_name]['directories'][method]

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = obs_name, avg_trials = avg_trials, clust_lb = clust_lb, calc_lb = calc_lb)

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadMNE(subID: str):

    MNEs = loadMNE(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions, with_std = True)

    return MNEs

# Build Spectrum Plotting iterable function
def it_spectrum(evoked_l: list):

    SP, E_SP = spectrum(evoked = evoked_l[0], s_evoked = evoked_l[1], ch_list = ch_list, N = N, wf = wf, window = window)

    return SP, E_SP

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

    print('\nDONE!')

    return MNEs_iters, points

# Build spectrum Plotting multiprocessing function
def mp_spectrum(evoks_iters: list, points: list):

    print('\nComputing Fourier Transform over each trial')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        results_ = list(tqdm(p.imap(it_spectrum, evoks_iters, chunksize = chunksize),
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

    lenght = len(freqs)

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),lenght]

    SP = collapse_trials(results = results, points = points, fshape = fshape, dtype = np.float64, e_results = e_results)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *SP)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults common shape: ', SP[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', SP[c].shape[0])

    print('')

    return

# Launch script with 'python -m recurrence' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    FREQUENCY SPECTRUM PLOT SCRIPT')

    evoks_iters, points = mp_loadMNE()

    mp_spectrum(evoks_iters = evoks_iters, points = points)

