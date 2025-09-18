# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

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
N_trajectories = 150

# Average trajectories or keep them separated
avg_trajectories = True

if avg_trajectories == True:
    method = 'avg_data'
    N_trajectories = 1
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

    print('\nGenerating Noise')
    print('\nSpawning ' + str(workers) + ' processes...')

    # Build mock iterable
    N_it = [i for i in range(0, N_trajectories)]

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        TS  = list(tqdm(p.imap_unordered(it_noise, N_it),#, chunksize = chunksize),
                       desc = 'It\'s just noise',
                       unit = 'it',
                       total = N_trajectories,
                       leave = False,
                       dynamic_ncols = True)
                       )

    TS = np.asarray(TS)[:,np.newaxis]

    # Create mne.info header
    info = mne.create_info(['noise'], sfreq = f)

    # Initzialize Evokeds list
    evokeds = []

    # Average trajectories
    if avg_trajectories == True:

        # Apply Z-Score
        if z_score == True:

            TS = zscore(TS)
            TS = TS[0]

        ev = mne.EvokedArray(TS, info, nave = 1, comment = 'noise')
        
        evokeds.append(ev)

    # Keep each individual one
    else:

        # Apply Z-Score
        if z_score == True:

            TS = zscore(TS)

        for trj in TS:

            ev = mne.EvokedArray(trj, info, nave = 1, comment = 'noise')
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

    print('\n    NOISE SCRIPT')

    mp_noise()

