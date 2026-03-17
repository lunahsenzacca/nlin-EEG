# Usual suspects
import os
import mne

import numpy as np

from parallelizer import mp_wrapper

# Utility function for Z-Scoring
from core import zscore

# Utility function for observables directories
from core import obs_path

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 1
chunksize = 1

### SCRIPT PARAMETERS ###

# ID for save
subID = '000'

# Dataset name to add subject
exp_name = 'noise'

# Experiment reference for time scales (and other stuff?)
ref_name = 'bmasking'

# Number of trajectories with different initial conditions
N_trajectories = 1

# Average trajectories or keep them separated
avg_trajectories = True

if avg_trajectories == True:
    method = 'avg_data'
else:
    method = 'trl_data'

# Apply Z-Score
z_score = True

### LOAD DATASET DIRECTORIES AND INFOS ###

# Load timepoints and sampling frequency from reference
T = maind[ref_name]['T']
f = maind[ref_name]['f']

###########################

### COMPUTATION ###

# Build iterable trajectory generation
def it_noise(trajectory_idx: int):

    ts = np.random.normal(size = T)

    ts = (ts - ts.mean())/ts.std()

    return ts

# Build trajectory generation multiprocessing function
def mp_noise():

    print('\nGenerating Noise\n')

    # Build mock iterable
    N_it = [i for i in range(0, N_trajectories)]

    TS = mp_wrapper(function = it_noise, iterable = N_it, workers = workers, chunksize = chunksize,
                    description = 'It\'s just noise:',
                    transient = False)

    TS = np.asarray(TS)[:,np.newaxis]

    # Folder path for saved MNE objects
    sv_path = maind[exp_name]['directories'][method]

    # Create directory
    os.makedirs(sv_path, exist_ok = True)

    # File name
    sv_file = sv_path + subID + '_noise'

    # Create mne.info header
    info = mne.create_info(['noise'], sfreq = f)

    info['description'] = subID + '_noise'

    # Average trajectories
    if avg_trajectories == True:

        # Apply Z-Score
        if z_score == True:

            TS = zscore(TS)

        n_trials = TS.shape[0]

        # Just the average
        avg = TS.mean(axis = 0)

        # Standard error not deviation!
        std = TS.std(axis = 0)/np.sqrt(n_trials)

        ev = mne.EvokedArray(avg, info, nave = n_trials, kind = 'average', comment = 'noise', verbose = False)
        s_ev = mne.EvokedArray(std, info, nave = n_trials, kind = 'standard_error', comment = 'noise', verbose = False)

        ev.save(sv_file + '-ave.fif', overwrite = True, verbose = False)
        s_ev.save(sv_file + '-std-ave.fif', overwrite = True, verbose = False)

    # Keep each individual one
    else:

        # Apply Z-Score
        if z_score == True:

            TS = zscore(TS)

        ep = mne.EpochsArray(TS, info, verbose = False)

        ep.save(sv_file + '-epo.fif', overwrite = True, verbose = False)

    print('\nDONE!')

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    NOISE SCRIPT')

    mp_noise()

