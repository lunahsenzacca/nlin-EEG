import os

import warnings

import numpy as np

from scipy.stats import linregress

from multiprocessing import Pool

from functions import to_log

from tqdm import tqdm

### SCRIPT PARAMETERS ###

# Multiprocessing parameters
workers = os.cpu_count() - 2
chunksize = 1

# Results folder
path = '/home/lunis/Documents/nlin-EEG/BM_CS/'

### FIT PARAMETERS ###

# Interval of r of interest (in indexes)
vlim = (7,22)

# Correction for expected signal (needs an explanation)
alpha = -0.5

### DATA TRANSFORMATION TO LOG SCALE ###

# Load correlation sum values
CS = np.load(path + 'CSums.npy')
r = np.load(path + 'rvals.npy')

# Reduced shape
rshp = CS.shape[1:4]

# Get log values
log_CS, log_r = to_log(CS, r)

### SCRIPT FOR COMPUTATION ###

# Build iterable
itrs = [i for i in log_CS]

# Build iterable function (over subjects)
def it_fit(abcd):

    c = 0 
    slope = []
    errslope = []

    zscore = []

    for abc in abcd:
        for ab in abc:
            for i, a in enumerate(ab):

                # Expected slope for a stochastic signal

                m = i + 2 + alpha

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = linregress(x = log_r, y = a, alternative = 'greater', nan_policy = 'omit')

                slope.append(results.slope)
                errslope.append(results.stderr)

                if np.isnan(results.stderr) == True or results.stderr == 0:
                    c+=1
                    zscore.append(0)
                else:
                    z = (results.slope - m)/results.stderr
                    zscore.append(z)

    slope = np.asarray(slope)
    errslope = np.asarray(errslope)

    zscore = np.asarray(zscore)

    slope = slope.reshape(rshp)
    errslope = errslope.reshape(rshp)

    zscore = zscore.reshape(rshp)

    return slope, errslope, zscore, c

# Build multiprocessing function
def mp_fit():


    print('Spawning ' + str(workers) + ' processes...')
    with Pool(workers) as p:
        
        res = list(tqdm(p.imap(it_fit, itrs), #chunksize = chunksize),
                       desc = 'Computing subjects ',
                       unit = 'subs',
                       total = len(itrs),
                       leave = False)
                        )
    slope = []
    errslope = []
    zscore = []
    c = 0
    for r in res:
        
        slope.append(r[0])
        errslope.append(r[1])
        zscore.append(r[2])
        c+=r[3]
    
    slope = np.asarray(slope)
    errslope = np.asarray(errslope)
    zscore = np.asarray(zscore)

    return slope, errslope, zscore, c

# launch script with 'python -m logfit' in appropriate conda enviroment
if __name__ == '__main__':

    # Compute results
    slope, errslope, zscore, c = mp_fit()
    print('DONE!')
    print('Number of failed regressions: ' + str(c))

    # Save results to local
    np.save(path + str(vlim[0]) + '_' + str(vlim[1]) + 'slopes.npy', slope)
    np.save(path + str(vlim[0]) + '_' + str(vlim[1]) + 'errslopes.npy', errslope)
    np.save(path + str(vlim[0]) + '_' + str(vlim[1]) + 'zscore.npy', zscore)
    print('Results shape: ', slope.shape)