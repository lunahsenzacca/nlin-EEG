# Usual suspects
import os
import json
import mne

import numpy as np

from tqdm import tqdm

# Sub-wise function for evoked file loading
from core import loadevokeds

# Evoked-wise function for Correlation Sum (CS) computation
from core import correlation_sum

# Utility function for observables directories
from core import obs_path

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###
workers = 20
chunksize = 1

### SCRIPT PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Label for results folder
lb = 'CPOF'

# Get data averaged across trials
avg_trials = False

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
sv_path = obs_path(exp_name = exp_name, obs_name = 'corrsum', clust_lb = lb, avg_trials = avg_trials)

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[0:1]
#ch_list = ch_list[0:2]

#Only averaged conditions
conditions = conditions[0:2]
#Parieto-Occipital and Frontal electrodes
ch_list = ['O2','PO4','PO8'],['Fp1','Fp2','Fpz']
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Embedding dimensions
embeddings = [i for i in range(2,21)]

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
                'embeddings' : embeddings
            }

### COMPUTATION ###

# Build evokeds loading iterable function
def it_loadevokeds(subID: str):

    evokeds = loadevokeds(subID = subID, exp_name = exp_name,
                          avg_trials = avg_trials, conditions = conditions)

    return evokeds


# Build Correlation Sum iterable function
def it_correlation_sum(evoked: mne.Evoked):

    CS = correlation_sum(evoked = evoked, ch_list = ch_list,
                        embeddings = embeddings, tau = tau, fraction = frc,
                        rvals = r)

    return CS

# Build evoked loading multiprocessing function
def mp_loadevokeds():

    print('\nPreparing evoked data')#\n\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:

        evoks = list(tqdm(p.imap(it_loadevokeds, sub_list), #chunksize = chunksize),
                       desc = 'Loading subjects ',
                       unit = 'sub',
                       total = len(sub_list),
                       leave = False)
                       )

    # Flatten and save separation points
    evoks_iters = [x for xss in evoks for xs in xss for x in xs]

    points = [[len(evoks[i][j]) for j in range(0,len(evoks[i]))] for i in range(0,len(evoks))]

    print('\nDONE!')

    return evoks_iters, points

# Build Correlation Sum multiprocessing function
def mp_correlation_sum(evoks_iters: list, points: list):

    complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)*len(r)

    velocity = 0.481

    import datetime
    eta = str(datetime.timedelta(seconds = int(complexity*velocity/workers)))

    print('\nComputing correlation sum over each trial')

    print('\nNumber of single computations: ' + str(complexity))

    print('\nEstimated completion time: ~' + eta)

    print('\nSpawning ' + str(workers) + ' processes...')
    
    if avg_trials == True:
        unit = 'sub'
    else:
        unit = 'trl'

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        res = list(tqdm(p.imap(it_correlation_sum, evoks_iters), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = unit,
                       total = len(evoks_iters),
                       leave = False)
                       )
    
    CS = []
    CS_STD = []
    count = 0
    for s in range(0,len(sub_list)):
        for c in range(0,len(conditions)):

            # Average across trial results
            avg = np.mean(np.asarray(res[count:count + points[s][c]]), axis = 0)
            std = np.std(np.asarray(res[count:count + points[s][c]]), axis = 0)

            CS.append(avg)
            CS_STD.append(std)

            count = count + points[s][c]

    CS = np.asarray(CS)
    CS_STD = np.asarray(CS_STD)

    CS = CS.reshape((len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r)))
    CS_STD = CS_STD.reshape((len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r)))

    CS = np.concatenate((CS[:,:,:,:,:,np.newaxis],CS_STD[:,:,:,:,:,np.newaxis]), axis = 5)
    
    print('\nDONE!')

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)

    np.save(sv_path + 'rvals.npy', r)
    np.save(sv_path + 'corrsum.npy', CS)

    variables['shape'] = CS.shape

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables,f)

    print('\nResults shape: ', CS.shape, '\n')

    return

# Launch script with 'python -m corrsum' in appropriate conda enviroment
if __name__ == '__main__':

    print('\n    CORRELATION SUM SCRIPT')

    evoks_iters, points = mp_loadevokeds()

    mp_correlation_sum(evoks_iters = evoks_iters, points = points)

