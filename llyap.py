# Usual suspects
import os
import json

import numpy as np

from tqdm import tqdm

# Sub-wise function for Largest Lyapunov Exponent (LLE) computation
from core import lyapunov

# Utility function for observables directories
from core import obs_path

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 2
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Label for results folder
lb = 'G'

### LOAD DATASET DIRECTORIES AND INFOS ###

# Raw folder path (yet to be implemented)
#path = maind[exp_name]['directories']['rw_data']

# Evoked folder path
path = maind[exp_name]['directories']['ev_data']

# List of ALL subject IDs
sub_list = maind[exp_name]['subIDs']

# List of ALL conditions
conditions = list(maind[exp_name]['conditions'].values())

# List of ALL electrodes
ch_list = maind[exp_name]['pois']

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'llyap', res_lb = lb)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[0:3]
#ch_list = ch_list[0:3]

#Only averaged conditions
conditions = list(conditions)[0:2]
#Parieto-Occipital and Frontal electrodes
#ch_list = ['Fp1','Fp2','Fpz']#['O2','PO4','PO8'],
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Embedding dimensions
embs = [2,3,4,5,6,7,8,9,10]#[11,12,13,14,15,16,17,18,19,20]#

# Time delay
tau = maind[exp_name]['tau']

# Window of interest
frc = [0, 1]

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clt = True
else:
    clt = False

# Dictionary for computation variables
variables = {   
                'tau' : tau,
                'window' : frc,
                'clustered' : clt,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'embeddings' : embs
            }

### SCRIPT FOR COMPUTATION ###

# Build iterable function
def it_lyapunov(subID: str):

    ly = lyapunov(subID = subID, conditions = conditions, ch_list = ch_list,
                                 embeddings = embs, tau = tau, fraction = frc,
                                 pth = path, verbose = False)
    return ly

# Build multiprocessing function
def mp_lyapunov():

    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        res = list(tqdm(p.imap(it_lyapunov, sub_list), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'subs',
                       total = len(sub_list),
                       leave = True)
                       )

    return np.asarray(res)

# Launch script with 'python -m correlation' in appropriate conda enviroment
if __name__ == '__main__':

    # Compute results
    results = mp_lyapunov()

    print('\nDONE!\n')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'lyap.npy', results)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('Results shape: ', results.shape)

