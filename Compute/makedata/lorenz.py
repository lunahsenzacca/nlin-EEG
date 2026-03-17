# Usual suspects
import os
import mne

import numpy as np

from parallelizer import mp_wrapper

# Utility function for Z-Scoring
from core import zscore

# Utility function for lorenz attractor time series
from core import lorenz_trajectory

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 4
chunksize = 1

### SCRIPT PARAMETERS ###

# ID for save
subID = '002'

# Dataset name to add subject
exp_name = 'lorenz'

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
def it_lorenz_trajectory(trajectory_idx: int):

    ts = lorenz_trajectory(dt = 0.01, time_points = 1000, target_l = T)

    ts = ts.sum(axis = 0)

    return ts

# Build trajectory generation multiprocessing function
def mp_lorenz_trajectory():

    print('\nGenerating Lorenz trajectories\n')

    # Build mock iterable
    N_it = [i for i in range(0, N_trajectories)]

    TS = mp_wrapper(function = it_lorenz_trajectory, iterable = N_it, workers = workers, chunksize = chunksize,
                    description = 'Expect some turbulence:',
                    transient = False)

    TS = np.asarray(TS)[:,np.newaxis]

    # Folder path for saved MNE objects
    sv_path = maind[exp_name]['directories'][method]

    # Create directory
    os.makedirs(sv_path, exist_ok = True)

    # File name
    sv_file = sv_path + subID + '_lorenz'

    # Create mne.info header
    info = mne.create_info(['lorenz'], sfreq = f)

    info['description'] = subID + '_lorenz'

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

        ev = mne.EvokedArray(avg, info, nave = n_trials, kind = 'average', comment = 'lorenz', verbose = False)
        s_ev = mne.EvokedArray(std, info, nave = n_trials, kind = 'standard_error', comment = 'lorenz', verbose = False)

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

    print('\n    LORENZ ATTRACTOR SCRIPT')

    mp_lorenz_trajectory()

