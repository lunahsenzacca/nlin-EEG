# Usual suspects
import os
import json
import mne

import numpy as np

# Multiprocessing wrapper
from core import mp_wrapper

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
    clst = True
else:
    clst = False

# Get method string
if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### PARAMETERS FOR FREQUENCY SPECTRUM COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Number of signals to generate for frequency domain error estimation
N = parameters['N']

# Window factor for frequency space resolution
wf = parameters['wf']

# Load frequency domain informations and get freqencies array
exp_info, times = get_tinfo(exp_name = exp_name, avg_trials = avg_trials, window = window)

# Make a fake fft to get the same frequency binning
f_ts = np.zeros(len(times))

_, freqs = mne.time_frequency.psd_array_welch(f_ts,
                    sfreq = exp_info['sfreq'],  # Sampling frequency from the evoked data
                    fmin = exp_info['highpass'], fmax = exp_info['lowpass'],  # Focus on the filter range
                    n_fft = int(len(f_ts)*wf),  # Length of FFT (controls frequency resolution)
                    n_per_seg = int(len(f_ts)/wf),
                    verbose = False)

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'N': N,
                'window_factor': wf,
                'freqs': list(freqs),

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
                   with_std = True)

    return MNEs

# Build Spectrum Plotting iterable function
def it_spectrum(MNE_l: list):

    SP, E_SP = spectrum(MNE = MNE_l[0], sMNE = MNE_l[1], ch_list = ch_list, N = N, wf = wf, window = window)

    return SP, E_SP

# Build evoked loading multiprocessing function
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

# Build spectrum multiprocessing function
def mp_spectrum(MNEs_iters: list, points: list):

    print('\nComputing Fourier Transform over each trial')
    print('\nSpawning ' + str(workers) + ' processes...')

    results_ = mp_wrapper(it_spectrum, iterable = MNEs_iters,
                          workers = workers,
                          chunksize = chunksize,
                          desc = 'Computing',
                          unit = 'trl')

    results = []
    e_results = []
    for r in results_:

        results.append(r[0])
        e_results.append(r[1])

    lenght = len(freqs)

    print(len(results),len(results[0]),len(results[0][0]))

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),lenght]

    SP = collapse_trials(results = results, points = points, fshape = fshape, dtype = np.float64, e_results = e_results)

    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{obs_name}.npz', *SP)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables, f, indent = 2)

    with open(sv_path + 'info.json','w') as f:
        json.dump(info, f, indent = 2)

    print('\nResults common shape: ', SP[0].shape[1:])

    if avg_trials == False:

        print('\nTrials\n')
    
        for c, prod in enumerate([i + '_' + j for i in sub_list for j in conditions]):
            print(f'{prod}: ', SP[c].shape[0])

    print('')

    return

if __name__ == '__main__':

    print('\n    FREQUENCY SPECTRUM PLOT SCRIPT')

    MNEs_iters, points = mp_loadMNE()

    mp_spectrum(MNEs_iters = MNEs_iters, points = points)

