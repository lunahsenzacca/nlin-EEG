# Usual suspects
import os
import numpy as np

from tqdm import tqdm

# MNE library for evoked file format use
import mne

# Sub-wise function for MNE file type conversion
from core import toMNE

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###
workers = 5
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking_dense'

# Average across trials
avg_trials = True

# Subject IDs
sub_list = maind[exp_name]['subIDs']

# Condition IDs
conditions = list(maind[exp_name]['conditions'].values())

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

os.makedirs(sv_path, exist_ok = True)

# Build iterable function
def it_toMNE(subID: str):
    
    toMNE(subID = subID, exp_name = exp_name, avg_trials = avg_trials, z_score = z_score, baseline = baseline, sv_path = sv_path)
    
    return

# Build multiprocessing function
def mp_toMNE():

    print('\nConverting dataset to MNE file format')

    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        list(tqdm(p.imap(it_toMNE, sub_list), 
                            desc = 'Processing',
                            unit = 'sub',
                            total = len(sub_list),
                            leave = True,
                            dynamic_ncols = True))

    print('\nDONE!\n')

    return

# Main method, launch using 'python -m toMNE' in the appropriate enviroment
if __name__ == '__main__':

    mp_toMNE()
