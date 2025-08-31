import os

import warnings

import numpy as np

from scipy.stats import linregress

from multiprocessing import Pool

from functions import to_log

from tqdm import tqdm

### LOAD PARAMETERS ###

# Multiprocessing parameters
workers = os.cpu_count() - 2
chunksize = 1

# Load results folder
path = '/home/lunis/Documents/nlin-EEG/BM_CS/'

# Label for load results files
lb = 'G'

### FIT PARAMETERS ###

# Average correlation sum over electrodes
avg = True

# Set label for results
if avg == True:
    a_lb = 'avg'
else:
    a_lb = ''

# Interval of r of interest (in indexes)
vlim = (5,28)

### SAVE FILES PARAMETERS ###

# Save results folder
sv_path = path + lb + '/D2/'

# Label for results file
sv_lb = a_lb + '[' + str(vlim[0]) + '_' + str(vlim[1]) + ']'

### DATA TRANSFORMATION TO LOG SCALE ###

# Load correlation sum values
CS = np.load(path + lb + '/CSums.npy')
r = np.load(path + lb + '/rvals.npy')

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
def it_fit(abcd):

    c = 0 
    slope = []
    errslope = []

    intercept = []
    errintercept = []

    for abc in abcd:
        for ab in abc:
            for i, a in enumerate(ab):

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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

    return slope, errslope, c #, intercept, errintercept

# launch script with 'python -m logfit' in appropriate conda enviroment
if __name__ == '__main__':

    # Compute results
    slope, errslope, c = mp_fit()
    print('DONE!')
    print('Number of failed regressions: ' + str(c))

    # Save results to local
    os.makedirs(sv_path, exist_ok = True)
    np.save(sv_path + sv_lb + 'slopes.npy', slope)
    np.save(sv_path + sv_lb + 'errslopes.npy', errslope)
    #np.save(sv_path + sv_lb + 'intercept.npy', intercept)
    #np.save(sv_path + sv_lb + 'errintercept.npy', errintercept)
    print('Results shape: ', slope.shape)