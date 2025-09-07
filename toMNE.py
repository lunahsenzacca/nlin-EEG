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
exp_name = 'bmasking'

# Method for conversion
# Average across trials
method = 'avg_data'
# Keep each trial
#method = 'trl_data'

# Subject IDs
sub_list = maind[exp_name]['subIDs']

# Directory for saved results
sv_path = maind[exp_name]['directories'][method]

# Build iterable function
def it_toevoked(subID: str):

    evokeds = toevoked(subID = subID, exp_name = exp_name, method = method)

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
    print('\nSaving results...')
    os.makedirs(sv_path, exist_ok = True)
    for i in range(0,len(evokeds)):

        data = evokeds[i]

        mne.write_evokeds(sv_path + sub_list[i] + '-ave.fif', data, overwrite = True, verbose = False)

    print('\nDONE!\n')
    
    return

# Main method, launch using 'python -m toMNE' in the appropriate enviroment
if __name__ == '__main__':

    mp_toevoked()