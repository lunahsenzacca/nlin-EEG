# Usual suspects
import os
import numpy as np

from tqdm import tqdm

# MNE for time series format standardization and scipy for .mat file reading
import mne
import scipy.io as sio 

# Some helper functions
from core import sub_path, name_toidx

# Multiprocessing Pool
from multiprocessing import Pool

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### SCRIPT PARAMETERS ###

exp_name = 'bmasking'

# Get dataset info to embedd in MNE files
Tst = maind[exp_name]['T']
freq = maind[exp_name]['f']
conditions = maind[exp_name]['conditions'].values()
    
# Function for single subject conversion
def BM_toevoked(subID: str):

    # Navigate subject folder with raw .mat single trial files
    sub_folder = sub_path(subID, exp_name = exp_name)
    all_files = os.listdir(sub_folder)

    # Load the .mat file (Some folders could be missing it)
    mat_data = sio.loadmat(maind[exp_name]['directories']['ch_info'])

    # Get list of electrodes names in .mat file
    ch_list = []
    for m in mat_data['Channel'][0]:
        ch_list.append(str(m[0][0]))

    # Create info file for MNE
    ch_types = ['eeg' for n in range(0, len(ch_list))]

    inf = mne.create_info(ch_list, ch_types = ch_types, sfreq = freq)
    inf.set_montage(maind[exp_name]['montage'])

    # Initzialize evoked list
    evoked = []

    # Loop over conditions
    for cond in conditions:
        
        my_cond_files = [f for f in all_files if cond in f ]
        
        # Get indexes of electrodes
        untup = name_toidx(ch_list, exp_name = exp_name)

        all_trials = np.empty((0,len(untup),Tst))
            
        for f in my_cond_files:

            path = sub_folder+f
                
            mat_data = sio.loadmat(path)
                
            data = mat_data['F'][untup]

            all_trials = np.concatenate((all_trials, data[np.newaxis,:]), axis = 0)
        
        n_trials = all_trials.shape[0]
        signals = all_trials.mean(axis = 0)

        # Append new evoked file to list
        evoked.append(mne.EvokedArray(signals, inf, nave = n_trials, comment = cond))

    return evoked

# Build main method function

# Multiprocessing parameters
workers = os.cpu_count() - 4
chunksize = 1

# Subject IDs
sub_list = maind[exp_name]['subIDs']

# Save folder
sv_path = maind[exp_name]['directories']['ev_data']
os.makedirs(sv_path, exist_ok = True)

def BMtoevoked():
    with Pool(workers) as p:
        datas = list(tqdm(p.imap(BM_toevoked, sub_list), 
                    total = len(sub_list),
                    desc = 'Processing',
                    leave = True))

        for i, data in enumerate(datas):

            mne.write_evokeds( sv_path + sub_list[i] + '-ave.fif', data, overwrite = True, verbose = False)

    return

# Main method, launch using python -m BMtoMNE in the appropriate enviroment
if __name__ == '__main__':

    print('\nConverting Backward Masking dataset to MNE Evoked file formats...\n')
    BMtoevoked()
    print('\nDONE!\n')