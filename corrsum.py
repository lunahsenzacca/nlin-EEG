# Usual suspects
import os
import json

import numpy as np

from tqdm import tqdm

# Sub-wise function for Correlation Sum (CS) computation
from core import correlation_sum

# Utility function for observables directories
from core import obs_path

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###
workers = os.cpu_count() - 6
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Label for results folder
lb = 'G20'

# Get data averaged across trials
avg_trials = True

if avg_trials == True:
    method = 'avg_data'
else:
    method = 'trl_data'

### LOAD DATASET DIRECTORIES AND INFOS ###

# Evoked folder paths
path = maind[exp_name]['directories'][method]

# List of ALL subject IDs
sub_list = maind[exp_name]['subIDs']

# List of ALL conditions
conditions = list(maind[exp_name]['conditions'].values())

# List of ALL electrodes
ch_list = maind[exp_name]['pois']

# Directory for saved results
sv_path = obs_path(exp_name = exp_name, obs_name = 'corrsum', res_lb = lb, avg_trials = avg_trials)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[0:3]
#ch_list = ch_list[0:3]

#Only averaged conditions
conditions = conditions[0:2]
#Parieto-Occipital and Frontal electrodes
#ch_list = ['O2','PO4','PO8'],['Fp1','Fp2','Fpz']
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Embedding dimensions
embs = [i for i in range(2,21)]

# Time delay
tau = maind[exp_name]['tau']

# Window of interest
frc = [0, 1]

# Distances for sampling the dependance
r = np.logspace(0, 1.7, num = 20, base = 10)
r = r/1e7

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

### COMPUTATION ###

# Build iterable function
def it_correlation_sum(subID: str):

    CD = correlation_sum(subID = subID, conditions = conditions, ch_list = ch_list,
                        embeddings = embs, tau = tau, fraction = frc,
                        rvals = r, pth = path)
    return CD

# Build multiprocessing function
def mp_correlation_sum():

    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        res = list(tqdm(p.imap(it_correlation_sum, sub_list), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'subs',
                       total = len(sub_list),
                       leave = True)
                       )
   
    res = np.asarray(res)
    print('\nDONE!\n')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'rvals.npy', r)
    np.save(sv_path + 'CSums.npy', res)

    variables['shape0'] = res.shape

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('Results shape: ', res.shape, '\n')

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    results = mp_correlation_sum()

