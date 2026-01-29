# Usual suspects
import json
import mne

import numpy as np

# Frequency Spectrum computation functions
from modules.spectrum import spectrum

# Utility function for dimensional time and frequency domain of the experiment
from core import get_tinfo

# Multiprocessing wrappers
from parallelizer import loader, calculator

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'spectrum'

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

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(freqs)]

# Launch script with 'python -m spectrum'
if __name__ == '__main__':

    print('\n    FREQUENCY SPECTRUM PLOT SCRIPT')

    MNEs_iters, points = loader(info = info, with_std = True)

    calculator(spectrum.it_spectrum(info = info, parameters = parameters),
               MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = True)
