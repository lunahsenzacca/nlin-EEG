import os

import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from functions import correlation_sum

### SCRIPT PARAMETERS ###

# Multiprocessing parameters
workers = os.cpu_count() - 2
chunksize = 1

# Save folder for results
sv_pth = '/home/lunis/Documents/nlin-EEG/BM_CS/'

# Label for results files
lb = 'G'

### EXPERIMENT FOLDER AND INFOS ###

# Evoked folder path
path = '/home/lunis/Documents/nlin-EEG/BM_evoked/'

# List of subject IDs
sub_list = ['001','002','003','004','005','006','007','008','009','010',
            '011','012','013','014','015','016','017','018','019','020',
            '023','024','025','026','028','029','030',
            '031','033','034','035','036','037','038','040',
            '042']

# List of conditions
conditions = ['S__','S_1',
              'S__1', 'S__2', 'S__3', 'S__4',
              'S_11', 'S_12', 'S_13', 'S_14']

# List of electrodes
ch_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
           'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Iz',
           'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
           'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3',
           'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5',
           'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
           'Fpz', 'CPz', 'POz', 'Oz']

### FOR QUICKER EXECUTION ###
#sub_list = sub_list[0:3]
#ch_list = ch_list[0:3]

#Only averaged conditions
conditions = conditions[0:2]
#Parieto-Occipital and Frontal electrodes
#ch_list = ['O2','PO4','PO8','Fp1','Fp2','Fpz']
###########################

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Embedding dimensions
embs = [2,3,4,5,6,7,8,9,10]

# Time delay
tau = 20

# Window of interest
frc = [0, 1]

# Distances for sampling the dependance
r = np.logspace(0, 1.7, num = 20, base = 10)
r = r/1e7

### SCRIPT FOR COMPUTATION ###

# Build iterable function
def it_correlation_sum(subID: str):

    CD = correlation_sum(subID = subID, conditions = conditions, ch_list = ch_list,
                          embeddings = embs, tau = tau, fraction = frc,
                          rvals = r, pth = path)
    return CD

# Build multiprocessing function
def mp_correlation_sum():

    print('Spawning ' + str(workers) + ' processes...')
    with Pool(workers) as p:
        
        res = list(tqdm(p.imap(it_correlation_sum, sub_list), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'subs',
                       total = len(sub_list),
                       leave = False)
                       )

    return np.asarray(res)

# Launch script with 'python -m correlation' in appropriate conda enviroment
if __name__ == '__main__':

    # Compute results
    results = mp_correlation_sum()

    print('DONE!')

    # Save results to local
    os.makedirs(sv_pth, exist_ok = True)

    np.save(sv_pth + lb + 'rvals.npy', r)
    np.save(sv_pth + lb + 'CSums.npy', results)

    print('Results shape: ', results.shape)

