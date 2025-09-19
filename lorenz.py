# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

# Utility function for Z-Scoring
from core import zscore

# Utility function for lorenz attractor time series
from core import lorenz_trajectory

# Utility function for observables directories
from core import obs_path

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 4
chunksize = 1

### SCRIPT PARAMETERS ###

# ID for save
subID = '000'

# Dataset name to add subject
exp_name = 'lorenz'

# Experiment reference for time scales (and other stuff?)
ref_name = 'bmasking'

# Number of trajectories with different initial conditions
N_trajectories = 150

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

    print('\nGenerating Lorenz trajectories')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Build mock iterable
    N_it = [i for i in range(0, N_trajectories)]

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        TS  = list(tqdm(p.imap_unordered(it_lorenz_trajectory, N_it),#, chunksize = chunksize),
                       desc = 'Evolving initial conditions',
                       unit = 'it',
                       total = N_trajectories,
                       leave = False,
                       dynamic_ncols = True)
                       )

    TS = np.asarray(TS)[:,np.newaxis]

    # Create mne.info header
    info = mne.create_info(['lorenz'], sfreq = f)

    # Initzialize Evokeds list
    evokeds = []

    # Average trajectories
    if avg_trajectories == True:

        n_trials = len(TS)
        avg = TS.mean(axis = 0)

        # Apply Z-Score
        if z_score == True:

            avg = zscore(avg[np.newaxis])
            avg = avg[0]

        ev = mne.EvokedArray(avg, info, nave = n_trials, comment = 'lorenz')
        evokeds.append(ev)

    # Keep each individual one
    else:

        # Apply Z-Score
        if z_score == True:

            TS = zscore(TS)

        for trj in TS:

            ev = mne.EvokedArray(trj, info, nave = 1, comment = 'lorenz')
            evokeds.append(ev)

    print('\nDONE!')

    # Evoked folder path for saved subject
    sv_path = maind[exp_name]['directories'][method]

    # Create directory
    os.makedirs(sv_path, exist_ok = True)

    # Evoked file directory
    sv_path = sv_path + subID + '-ave.fif'

    mne.write_evokeds(sv_path, evokeds, overwrite = True, verbose = False)

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    LORENZ ATTRACTOR SCRIPT')

    mp_lorenz_trajectory()

