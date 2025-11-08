# Usual suspects
import os
import numpy as np

from tqdm import tqdm

# MNE library for evoked file format use
import mne

# Sub-wise function for evoked conversion
from core import toevoked

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###
workers = 10
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking_dense'

# Average across trials
avg_trials = True

# Subject IDs
sub_list = maind[exp_name]['subIDs']

# Apply Z-Score
z_score = True

# Apply whole trial baseline
baseline = False

if z_score == True:
    sv_name = 'z' + exp_name
else:
    sv_name = exp_name

# Directory for saved results
if avg_trials == True:
    sv_path = maind[sv_name]['directories']['avg_data']
else:
    sv_path = maind[sv_name]['directories']['trl_data']

# Build iterable function
def it_toevoked(subID: str):

    evokeds = toevoked(subID = subID, exp_name = exp_name, avg_trials = avg_trials, z_score = z_score, baseline = baseline)

    return evokeds

# Build multiprocessing function
def mp_toevoked():

    print('\nConverting dataset to MNE Evoked file format')

    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        evokeds = list(tqdm(p.imap(it_toevoked, sub_list), 
                    desc = 'Processing',
                    unit = 'sub',
                    total = len(sub_list),
                    leave = True,
                    dynamic_ncols = True)
                    )

    # Save results to local
    print('\nSaving evokeds...')

    os.makedirs(sv_path, exist_ok = True)
    
    for i in range(0,len(evokeds)):

        if avg_trials == False:

            data = []
            for c_evoked in evokeds[i]:

                data.append(c_evoked[0])

            mne.write_evokeds(sv_path + sub_list[i] + '-ave.fif', data, overwrite = True, verbose = False)

        else:

            data = []
            std = []
            for c_evoked in evokeds[i]:

                data.append(c_evoked[0])
                std.append(c_evoked[1])

            mne.write_evokeds(sv_path + sub_list[i] + '-ave.fif', data, overwrite = True, verbose = False)
            mne.write_evokeds(sv_path + sub_list[i] + '_std-ave.fif', std, overwrite = True, verbose = False)

    print('\nDONE!\n')
    
    return

# Main method, launch using 'python -m toMNE' in the appropriate enviroment
if __name__ == '__main__':

    mp_toevoked()