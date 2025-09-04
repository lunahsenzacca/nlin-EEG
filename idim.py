# Usual suspects
import os
import json
import warnings

import numpy as np

from tqdm import tqdm

# Scipy function for linear regression
from scipy.stats import linregress

# Utility function for log conversion
from core import to_log

# Utility function for observables directories
from core import obs_path

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSIN PARAMETERS ###

workers = 10
chunksize = 1

### LOAD PARAMETERS ###

# Dataset name
exp_name = 'bmasking'

# Label for load results files
lb = 'G'

# Label for saved results files
sv_lb = 'GoodRange'

# Correlation Sum results directory
path = obs_path(exp_name = exp_name, obs_name = 'corrsum', res_lb = lb)

# D2 saved results directory
sv_path = obs_path(exp_name = exp_name, obs_name = 'idim', res_lb = lb, calc_lb = sv_lb)

### FIT PARAMETERS ###

# Average correlation sum over electrodes
avg = False

# Interval of r of interest (in indexes)
vlim = (9,19)

### RESULTS DICTIONARY ###

# Load result variables
with open(path + 'variables.json', 'r') as f:
    variables = json.load(f)

clst = variables['clustered']

# Deny average method when corrsum results come from clustered pois
if clst == True:
    avg = False
    print('\nClustered input: \'avg\' variable bypassed to \'False\'')

# Load correlation sum values
CS = np.load(path + 'CSums.npy')
r = np.load(path + 'rvals.npy')

# Add entries to dictionary for save results
variables['vlim'] = vlim
variables['avg'] = avg
variables['shape0'] = CS.shape
variables['shape1'] = CS.shape[:-2]

### DATA TRANSFORMATION TO LOG SCALE ###

if avg == True:
    CS = CS.mean(axis = 2)[:,:,np.newaxis,:,:]

# Reduced shape
rshp = CS.shape[1:4]

# Get log values
log_CS, log_r = to_log(CS, r)

### SCRIPT FOR COMPUTATION ###

# Build iterable
itrs = [i for i in log_CS]

# Build iterable function (over subjects)
def it_fit(iinlog_CS: np.ndarray):

    # Bad regression counter
    c = 0 

    # Initzialize results arrays
    slope = []
    errslope = []

    # Intercept results are easily attached
    #intercept = []
    #errintercept = []

    abcd = iinlog_CS
    for abc in abcd:
        for ab in abc:
            for i, a in enumerate(ab):

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    results = linregress(x = log_r, y = a, alternative = 'greater', nan_policy = 'omit')

                slope.append(results.slope)
                errslope.append(results.stderr)
                #intercept.append(results.intercept)
                #errintercept.append(results.intercept_stderr)

                if np.isnan(results.stderr) == True: #or results.stderr == 0:
                    c+=1

    slope = np.asarray(slope)
    errslope = np.asarray(errslope)
    #intercept = np.asarray(intercept)
    #errintercept = np.asarray(errintercept)

    slope = slope.reshape(rshp)
    errslope = errslope.reshape(rshp)
    #intercept = intercept.reshape(rshp)
    #errintercept = errintercept.reshape(rshp)

    return slope, errslope, c #,intercept, errintercept

# Build multiprocessing function
def mp_fit():

    print('\nSpawning ' + str(workers) + ' processes...')

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        res = list(tqdm(p.imap(it_fit, itrs), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'subs',
                       total = len(itrs),
                       leave = False)
                        )
    slope = []
    errslope = []
    #intercept = []
    #errintercept = []
    c = 0
    for r in res:
        
        slope.append(r[0])
        errslope.append(r[1])
        c+=r[2]
        #intercept.append(r[3])
        #errintercept.append(r[4])

    
    slope = np.asarray(slope)
    errslope = np.asarray(errslope)
    #intercept = np.asarray(intercept)
    #errintercept = np.asarray(errintercept)

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)
    np.save(sv_path + 'slopes.npy', slope)
    np.save(sv_path + 'errslopes.npy', errslope)
    #np.save(sv_path + 'intercept.npy', intercept)
    #np.save(sv_path + 'errintercept.npy', errintercept)

    with open(sv_path + 'variables.json', 'w') as f:
        json.dump(variables, f)

    return slope, errslope, c #, intercept, errintercept


# Launch script with 'python -m idim' in appropriate conda enviroment
if __name__ == '__main__':

    # Compute results
    slope, errslope, c = mp_fit()
    print('\nDONE!\n')
    print('Number of bad regressions: ' + str(c))

    print('\nResults shape: ', slope.shape, '\n')