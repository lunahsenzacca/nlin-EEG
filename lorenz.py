# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

# Utility function for lorenz attractor time series
from core import lorenz_trajectory

# Utility function for observables directories
from core import obs_path

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###
workers = 16
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

# Method for reduction
#reduction = 'sum'
#reduction = 'mean'

# Average trajectories or keep them separated
avg_trajectories = True

if avg_trajectories == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### LOAD DATASET DIRECTORIES AND INFOS ###

# Load timepoints and sampling frequency from reference
T = maind[ref_name]['T']
f = maind[ref_name]['f']

###########################

### COMPUTATION ###

# Build iterable trajectory generation
def it_lorenz_trajectory(trajectory_idx: int):

    ts = lorenz_trajectory(dt = 0.0005, time_points = 7000, target_l = T)

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
                       desc = 'Loading subjects ',
                       unit = 'itr',
                       total = N_trajectories,
                       leave = False,
                       dynamic_ncols = True)
                       )

    TS = np.asarray(TS)

    # Create mne.info header
    info = mne.create_info(['lorenz'], sfreq = f)

    # Initzialize Evokeds list
    evokeds = []

    # Average trajectories
    if avg_trajectories == True:

        n_trials = len(TS)
        avg = TS.mean(axis = 0)

        avg = avg[np.newaxis]

        ev = mne.EvokedArray(avg, info, nave = n_trials, comment = 'lorenz')
        evokeds.append(ev)

    # Keep each individual one
    else:

        for trj in TS:

            trj = trj[np.newaxis]

            ev = mne.EvokedArray(trj, info, nave = 1, comment = 'lorenz')
            evokeds.append(ev)

    print('\nDONE!')

    # Evoked folder path for saved subject
    sv_path = maind[exp_name]['directories'][method]

    # Create directory
    os.makedirs(sv_path, exist_ok = True)

    # Evoked file directory
    sv_path = sv_path + subID + '-ave.fif'

    mne.write_evokeds(sv_path, evokeds, overwrite = True, verbose = True)

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    LORENZ ATTRACTOR SCRIPT')

    mp_lorenz_trajectory()

