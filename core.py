'''
In this file we define the various functions for dataset navigation and
analysis.

EDIT THE init.py FILE ACCORDINGLY BEFORE AND AFTER YOU MAKE CHANGES HERE.
'''
# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Usual suspects
import os
import mne
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

    # Teaspoon library functions #

# Autocorrelation time according to first minimum of Mutual Information (FMMI)
from teaspoon.parameter_selection.MI_delay import MI_for_delay

# Autocorrelation time according to first minimum of Mutual Information (FMMI)
from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau

# False Nearest Neighbour method
from teaspoon.parameter_selection.FNN_n import FNN_n

# Permutation Entropy (Not Used For Now...)
from teaspoon.SP.information.entropy import PE

# Zero Dimensional Sublevel Set Persistance Diagrams
from teaspoon.TDA.SLSP import Persistence0D

# Compute cutoff for Persistence Features
from teaspoon.TDA.SLSP_tools import cutoff

    # Scipy library #

# Method for .mat file loading
import scipy.io as sio

# Linear regression
from scipy.stats import linregress

# Power spetrum of time series (Uses FFT)
from scipy.signal import periodogram

# Find peaks in data
from scipy.signal import find_peaks


### UTILITY FUNCTIONS ###

'''
sub_path         : Get subject rw_data directory;

obs_path         : Get observable directory;

obs_data         : Get observable data;

name_toidx       : Converts tuple or list of electrode names to corresponding .mat
                   file index, for bw dataset only for now;

tuplinator       : Old helper function for functions that do not use mne
                   data type (SSub_ntau, twodim_graphs);

raw_tolist       : Extract raw data and prepare it for MNE evoked file format conversion
                   by computing the input for list_toevoked;
                   
toinfo           : Create info file for MNE evoked file format;

list_toevoked    : Converts multiple trial array to MNE evoked file format;

toevoked         : Concatenates taw_tolist and list_toevoke;

loadevokeds       : Loads evoked files.
'''

# Cython modules compilation
def cython_compile(verbose: bool, setup_name = 'cython_setup'):

    print('\nCompiling cython file...')

    with warnings.catch_warnings():

        if verbose == True:

            warnings.simplefilter('ignore')

        os.system(f'python ./cython_modules/{setup_name}.py build_ext -b ./cython_modules/ -t ./cython_modules/build/')

    return

# Get subject path
def sub_path(subID: str, exp_name: str):

    # Text to attach before and after subID to get subject folder name for raw data
    snip = maind[exp_name]['directories']['subject']
    
    path = snip[0] + subID + snip[1]

    return path

# Get observable path
def obs_path(exp_name: str, obs_name: str, clust_lb: str, avg_trials: bool, calc_lb = None):

    if avg_trials == True:
        results = 'avg_results'
    else:
        results = 'trl_results'

    # Create directory string
    path = maind[exp_name]['directories'][results] + clust_lb + '/'+ maind['obs_lb'][obs_name] + '/'

    if calc_lb != None:

        path = path + calc_lb + '/'

    return path

# Get pics path
def pics_path(exp_name: str, obs_name: str, clust_lb: str, avg_trials: bool, calc_lb = None):

    if avg_trials == True:
        pics = 'avg_pics'
    else:
        pics = 'trl_pics'

    # Create directory string
    path = maind[exp_name]['directories'][pics] + clust_lb + '/'+ maind['obs_lb'][obs_name] + '/'

    if calc_lb != None:

        path = path + calc_lb + '/'

    return path

# Get observable data
def obs_data(obs_path: str, obs_name: str):

    # Load result variables
    with open(obs_path + 'variables.json', 'r') as f:
        variables = json.load(f)

    if obs_name == 'epochs':

        M = np.load(obs_path + 'epochs.npz')
        x = variables['t']

        X = [x]

    elif obs_name == 'spectrum':

        M = np.load(obs_path + 'spectrum.npz')
        x = variables['freqs']

        X = [x]

    elif obs_name == 'delay':

        M = np.load(obs_path + 'delay.npz')
        x = variables['pois']

        X = [x]

    elif obs_name == 'recurrence':

        M = np.load(obs_path + 'recurrence.npz')
        x = variables['embeddings']
        y = variables['log_r']

        X = [x,y]

    elif obs_name == 'separation':

        M = np.load(obs_path + 'separation.npz')
        x = variables['embeddings']
        y = variables['dt']

        X = [x,y]

    elif obs_name == 'corrsum':

        M = np.load(obs_path + 'corrsum.npz')
        x = variables['embeddings']
        y = variables['log_r']

        X = [x,y]

    elif obs_name == 'correxp':

        M = np.load(obs_path + 'correxp.npz')
        x = variables['embeddings']
        y = variables['log_r']

        X = [x,y]

    elif obs_name == 'peaks':

        M = np.load(obs_path + 'peaks.npz')
        x = variables['embeddings']

        # See if you want to implement it later
        #y = np.load(obs_path + 'peaks_r.npz')

        X = [x]

    elif obs_name == 'plateaus':

        M = np.load(obs_path + 'plateaus.npz')
        x = variables['embeddings']
        y = variables['log_r']

        # See if you want to implement it later
        z = np.load(obs_path + 'plateaus_r.npz')

        X = [x,y,z]

    elif obs_name == 'idim':

        M = np.load(obs_path + 'idim.npz')
        x = variables['embeddings']

        X = [x]

    elif obs_name == 'llyap':

        M = np.load(obs_path + 'llyap.npz')
        x = variables['embeddings']

        X = [x]

    return M, X, variables

# Convert channel names to appropriate .mat data index
def name_toidx(names: list| tuple, exp_name: str):

    # Get list of electrodes names
    ch_list = maind[exp_name]['pois']

    # Check if we are clustering electrodes with tuples
    if type(names) ==  tuple:

        first = True
        for c in names:
            part = []
            
            for sc in c:
                
                # Add new indexes to cluster list
                part = part + [np.where(np.asarray(ch_list)==sc)[0][0]]

            part = part,
            if first == True:

                ch_clust_idx = part
                first = False

            else:
                ch_clust_idx = ch_clust_idx + part
                

    else:

        ch_clust_idx = []
        for c in names:
            ch_clust_idx.append(np.where(np.asarray(ch_list)==c)[0][0])

    return ch_clust_idx

# Convert a list into a tuple of lists
def tuplinator(list: list):

    tup = [],
    first = True
    for i in list:
        add = [i],

        if first == True:
            tup = add
            first = False
        else:
            tup = tup + add

    return tup

# Get time array for a specific crop
def get_tinfo(exp_name: str, method: str, fraction: list):

    # Evoked folder paths
    path = maind[exp_name]['directories'][method]

    # List of ALL subject IDs
    sub_list = maind[exp_name]['subIDs']

    # Load just one subject
    example_path = path + sub_list[0] + '-ave.fif'

    evoked = mne.read_evokeds(example_path, verbose = False)

    times = evoked[0].times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]
    
    evoked[0].crop(tmin = tmin, tmax = tmax, include_tmax = False)

    times = evoked[0].times

    info = evoked[0].info

    return info, times

# Function for single subject conversion from raw data to list  for MNE 
def raw_tolist(subID: str, exp_name: str):

    # Navigate subject folder with raw .mat single trial files
    sub_folder = sub_path(subID, exp_name = exp_name)
    all_files = os.listdir(sub_folder)

    # Get list of electrodes names in .mat file
    ch_list = maind[exp_name]['pois']

    # Get time points lenght
    Tst = maind[exp_name]['T']

    # Get conditions
    conditions = list(maind[exp_name]['conditions'].values())

    # Loop over conditions
    data_list = []
    for cond in conditions:

        my_cond_files = [f for f in all_files if cond in f ]

        # Get indexes of electrodes
        untup = name_toidx(ch_list, exp_name = exp_name)

        all_trials = np.empty((0,len(untup),Tst))
            
        for f in my_cond_files:

            path = sub_folder + f
            
            # This also depends on raw data structure
            if exp_name == 'bmasking' or exp_name == 'zbmasking':
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup][np.newaxis]

                # There is probably a way to fix this
                info = None

            else:

                mne_data = mne.read_epochs(path, preload = True, verbose = False)

                data = mne_data.get_data(picks = untup)

                info = mne_data.info

            all_trials = np.concatenate((all_trials, data), axis = 0)

        data_list.append(all_trials)

    return data_list, info

# Create info file for specific datased
def toinfo(exp_name: str, info: mne.Info, ch_type = 'eeg'):

    if type(info) == None:

        # Get electrodes labels
        ch_list = maind[exp_name]['pois']

        ch_types = [ch_type for n in range(0, len(ch_list))]

        # Get sampling frequency
        freq = maind[exp_name]['f']

        info = mne.create_info(ch_list, ch_types = ch_types, sfreq = freq)

    return info

# Function for subwise evoked conversion from list of arrays
def list_toevoked(data_list: list, info: mne.Info, subID: str, exp_name: str, avg_trials: bool, z_score: bool, baseline: bool, alt_sv: str):
    
    # There are different number of trials for each condition,
    # so a simple ndarray is inconvenient. We use a list of ndarray instead

    # The list index cycles faster around the conditions and slower around the subject

    # Create info file
    info = toinfo(exp_name = exp_name, info = info)

    conditions = list(maind[exp_name]['conditions'].values())

    # Initialize evokeds list
    evokeds = []

    # Cycle around conditions
    for i, array in enumerate(data_list):

        # 'array' structure
        # Axis 0 = Trials
        # Axis 1 = Electrodes
        # Axis 2 = Time points

        # Check that we are not being dumb
        if z_score == True and baseline == True:

            print('\nTrying to apply both baseline and zscore, not allowed!')

            return

        # Apply zscore normalization
        elif z_score == True:

            array_ = zscore(array)

        # Apply dumb baseline correction 
        elif baseline == True:

            array_ = array.copy() - np.broadcast_to(array.mean(axis = 2)[:,:,np.newaxis], array.shape)

        else:

            array_ = array.copy()

        # Compute average and standard error across trials
        if avg_trials == True:

            n_trials = array.shape[0]

            # Just the average
            avg = array_.mean(axis = 0)

            # Standard error not deviation!
            std = array_.std(axis = 0)/np.sqrt(n_trials)

            ev = mne.EvokedArray(avg, info, nave = n_trials, comment = conditions[i], kind = 'average')
            s_ev = mne.EvokedArray(std, info, nave = n_trials, comment = conditions[i], kind = 'standard_error')

            evokeds.append([ev, s_ev])

        # Or keep each individual trial
        else:

            for trl in array_:

                ev = mne.EvokedArray(trl, info, nave = 1, comment = conditions[i])
                evokeds.append([ev])

    # This is meant for testing in notebooks
    if alt_sv != None:

        # Create directory
        os.makedirs(alt_sv, exist_ok = True)

        # Evoked file directory
        sv_path = alt_sv + subID + '-ave.fif'

        mne.write_evokeds(sv_path, evokeds, overwrite = True, verbose = False)
 
    return evokeds

# Create evoked file straight from raw data
def toevoked(subID: str, exp_name: str, avg_trials: bool, z_score = False, baseline = False, alt_sv = None):

    # Create data list
    data_list, info = raw_tolist(subID = subID, exp_name = exp_name)

    # Generate evokeds
    evokeds = list_toevoked(data_list = data_list, info = info, subID = subID, exp_name = exp_name, avg_trials = avg_trials, z_score = z_score, baseline = baseline, alt_sv = alt_sv)

    return evokeds

# Load evoked files of a specific subject
def loadevokeds(exp_name: str, avg_trials: bool, subID: str, conditions: list, std = False):

    # Select correct path for data
    if avg_trials == True:
        file_p = maind[exp_name]['directories']['avg_data']
    else:
        file_p = maind[exp_name]['directories']['trl_data']

    if std == False:

        evokeds = mne.read_evokeds(file_p + subID + '-ave.fif', verbose = False)

    else:

        evokeds = mne.read_evokeds(file_p + subID + '_std-ave.fif', verbose = False)

    # Create evoked list
    full_evokeds = []
    for cond in conditions:

        c_evokeds = []
        for e in evokeds:
            if e.comment == cond:
                c_evokeds.append(e)
        
        full_evokeds.append(c_evokeds)

    return full_evokeds

# Create one dimensional list with list of evoked objects per subject per condition
def flat_evokeds(evokeds: list):

    # Flatten evokeds nested list
    flat = [x for xss in evokeds for xs in xss for x in xs]

    # Save separation coordinates for 'collapse_trials' functions
    points = [[len(evokeds[i][j]) for j in range(0,len(evokeds[i]))] for i in range(0,len(evokeds))]

    return flat, points

# Open results .npz file and convert it to nested list structure
def loadresults(obs_path: str, obs_name: str, X_transform = None):

    M, X, variables = obs_data(obs_path = obs_path, obs_name = obs_name)

    len_s = len(variables['subjects'])
    len_c = len(variables['conditions'])

    results = []
    for s in range(0, len_s):

        c_results = []
        for c in range(0, len_c):

            c_results.append(M[M.files[len_c*s + c]])

        results.append(c_results)

    if X_transform != None:

        N = X[X_transform]
        X_results = []
        for s in range(0, len_s):

            c_results = []
            for c in range(0, len_c):

                c_results.append(N[N.files[len_c*s + c]])

            X_results.append(c_results)

        X[X_transform] = X_results

    return results, X, variables

# Create one dimensional list of results per subject per condition
def flat_results(results: list):

    # Flatten results nested list
    flat = [x for xss in results for xs in xss for x in xs]

    # Save separation coordinates for 'collapse_trials' functions
    points = [[len(results[i][j]) for j in range(0,len(results[i]))] for i in range(0,len(results))]

    return flat, points

# Function for trials averaging and homogeneous array generation (works for already averaged trials as well)
def collapse_trials(results: list, points: list, fshape: list, e_results = None, dtype = np.float64):

    if e_results == None:
        print('\nNo observable error in input, zeros will be appended in results along trial error')

    # Initzialize list of homegenous arrays
    RES = []

    # Make homogeneous arrays for each subject
    count = 0
    for s in range(0,fshape[0]):
        for c in range(0,fshape[1]):

            shape = [len(results[count:count + points[s][c]])]

            trials = np.asarray(results[count:count + points[s][c]], dtype = dtype)

            # Get error from computation uncertainty
            if e_results != None:

                e_trials = np.asarray(e_results[count:count + points[s][c]], dtype = dtype)

            else:

                e_trials = np.zeros(trials.shape, dtype = dtype)

            shape_ = [*shape, *fshape[2:]]

            trials = trials.reshape(shape_)
            e_trials = e_trials.reshape(shape_)

            trials = np.concatenate((trials[:,np.newaxis], e_trials[:,np.newaxis]), axis = 1)

            count = count + points[s][c]

            RES.append(trials)

    return RES

# Prepare corrsum.py results for correxp.py script
def correxp_getcorrsum(path: str):

    # Load correlation sum results
    CS, _, variables = loadresults(obs_path = path, obs_name = 'corrsum', X_transform = None)

    flat_CS, points = flat_results(CS)

    # Initzialize trial-wise iterable
    log_CS_iters = []
    for arr in flat_CS:

        log_CS = to_log(arr, verbose = False)

        # Build trial-wise iterable
        log_CS_iters.append([log_CS[0],log_CS[1]])

    return log_CS_iters, points, variables

# Prepare correxp.py results for peaks.py script
def pp_getcorrexp(path: str):

    # Load correlation sum results
    CE, _, variables = loadresults(obs_path = path, obs_name = 'correxp', X_transform = None)

    flat_CE, points = flat_results(CE)

    # Initzialize trial-wise iterable
    CE_iters = []
    for arr in flat_CE:

        # Build trial-wise iterable
        CE_iters.append([arr[0],arr[1]])

    return CE_iters, points, variables

# Prepare recurrence.py results for corrsum.py script
def corrsum_getrecurrence(path: str):

    # Load correlation sum results
    RP, _, variables = loadresults(obs_path = path, obs_name = 'recurrence', X_transform = None)

    flat_RP, points = flat_results(RP)

    # Initzialize trial-wise iterable
    RP_iters = []
    for arr in flat_RP:

        # Build trial-wise iterable
        RP_iters.append([arr[0],arr[1]])

    return RP_iters, points, variables

### HELPER FUNCTIONS ###

# Euclidean distance
def dist(x: np.array, y: np.array, m_norm = False, m = None):

    d = np.sqrt(np.sum((x - y)**2, axis = 0))

    if m_norm == True and m != None:

        d = d/m

    return d

# Transform data in log scale (Useful for logarithmic fits)
def to_log(OBS: np.ndarray, verbose: bool):

    # Initzialize results
    log_OBS = OBS[0].copy()
    e_log_OBS = OBS[1].copy()

    # Get shape
    shp = log_OBS.shape

    # Zero values counter
    c = 0

    flat = log_OBS.flatten()
    e_flat = e_log_OBS.flatten()

    r = []
    e_r = []
    for i, o in enumerate(tqdm(flat, total = len(flat), 
                                     desc = 'Getting log values and errors',
                                     leave = False,
                                     disable = not(verbose))):

        if o == 0:
            c += 1
            r.append(np.nan)
            e_r.append(np.nan)
        else:
            r.append(np.log(o))
            e_r.append(e_flat[i]/o)

    if verbose == True:
        print('Zero valued data points: ' + str(c))

    log_OBS = np.reshape(np.asarray(r), shp)
    e_log_OBS = np.reshape(np.asarray(e_r), shp)

    log_OBS = np.concatenate((log_OBS[np.newaxis], e_log_OBS[np.newaxis]), axis = 0)

    return log_OBS

# Get average period (in time points) of a 1d trajectory through periodogram
def avg_period(ts: list):

    f, ps = periodogram(ts)
    avT = np.sum(ps)/np.sum(f*ps)

    return avT

# Get Lorenz attractor derivatives
def lorenz_delta(xyz, s=10, r=28, b=8/3):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return np.array([x_dot, y_dot, z_dot])

# Generate lorentz system trajectory
def lorenz_trajectory(dt: float, time_points: int, target_l = None, x0 = None):

    # Generate random initial values
    if x0 == None:
        x0 = np.random.normal(loc = 0, scale = 1, size = 3)

    xyzs = np.empty((time_points + 1, 3))  # Need one more for the initial values
    xyzs[0] = x0  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(time_points):
        xyzs[i + 1] = xyzs[i] + lorenz_delta(xyzs[i])*dt

    xyzs = np.swapaxes(xyzs, 0, 1)

    # Downsample if target lenght is given
    if target_l != None:
        dxyzs = []
        for i, x in enumerate(xyzs):
            
            dxyzs.append(downsample(x, target_l = target_l))

            xyzs = np.asarray(dxyzs)

    return xyzs

# Downsample timepoints of a timeseries averaging per window
def downsample(ts, target_l: int):

    initial_l = len(ts)

    # Find interpolation lenght

    window = int(initial_l/target_l)

    if window == 0:
        print('target lenght too short')
        return

    partial = int(initial_l/window)

    downsampled = []
    for i in range(0,partial):

        w = [ts[i*window+j] for j in range(0,window)]
        downsampled.append(np.asarray(w).mean())

    return np.asarray(downsampled)

# Z-Score normalization for multiple trials timeseries
def zscore(trials_array: np.ndarray, keep_relations = False):
    
    # 'trials_array' structure
    # Axis 0 = Trials
    # Axis 1 = Electrodes
    # Axis 2 = Time points

    z_trials_array = np.copy(trials_array)

    m = trials_array.mean(axis = (0,2))
    s = trials_array.std(axis = (0,2))

    if keep_relations == True:

        m = m.mean()
        s = s.mean()

        #Apply Z-Score
        z_trials_array = (trials_array - m)/std

    # Full electrode-wise Z-Score
    else:
        # Faster and more memory efficient
        M = np.broadcast_to(m[np.newaxis,:,np.newaxis], [trials_array.shape[0],len(m),trials_array.shape[2]])
        S = np.broadcast_to(s[np.newaxis,:,np.newaxis], [trials_array.shape[0],len(s),trials_array.shape[2]])

        '''
        # Slower and less memory efficient
        m = m[np.newaxis,:,np.newaxis]
        s = s[np.newaxis,:,np.newaxis]
        
        M = m
        S = s
        for i in range(1,trials_array.shape[2]):

            M = np.concatenate((M,m),axis = 2)
            S = np.concatenate((S,s),axis = 2)
        
        m = M
        s = S
        for i in range(1,trials_array.shape[0]):

            M = np.concatenate((M,m),axis = 0)
            S = np.concatenate((S,s),axis = 0)
        '''

        # Apply Z-Score
        z_trials_array = np.divide(trials_array - M, S)

    return z_trials_array

# 1-D function Adymensional Gaussian kernel convolution
def gauss_kernel(function: np.array, x: np.array, scale: float, cutoff: int, order: int):

    i_len = len(function)

    # Create convoluted array
    c_function = np.copy(function)
    c_x = np.copy(x)

    # Create redundant head and tails
    f_tail = np.array([function[0] for i in range(0,cutoff)])
    f_head = np.array([function[-1] for i in range(0,cutoff)])

    x_tail = np.array([x[0]  for i in range(0,cutoff)])
    x_head = np.array([x[-1] for i in range(0,cutoff)])

    function = np.concatenate((f_tail, function, f_head))
    x = np.concatenate((x_tail, x, x_head))

    for i in range(0,i_len):

        vals = np.array([function[j] for j in range(i, 2*cutoff + i + 1)])
        if order == 0:
            ker = np.array([np.exp((x[j]-c_x[i])**2)/(2*(scale**2)) for j in range(i, 2*cutoff + i + 1)])
        elif order == 1:
            print('Order 1 kernel Not yet implemented')
            return

        c_function[i] = np.sum(vals*ker)/np.sum(ker)

    return c_function


### TIME SERIES MANIPULATION FUNCTIONS ###

# Time-delay embedding of a single time series
def td_embedding(ts: np.array, embedding: int, tau: int):

    min_len = (embedding - 1)*tau + 1

    # Check if embedding is possible
    if len(ts) < min_len:

        print('Data lenght is insufficient, try smaller parameters')
        return
    
    # Set lenght of embedding
    m = len(ts) - min_len + 1

    # Get indexes
    idxs = np.repeat([np.arange(embedding)*tau], m, axis = 0)
    idxs += np.arange(m).reshape((m, 1))

    emb_ts = ts[idxs]

    emb_ts = np.asarray(emb_ts, dtype = np.float64)

    return emb_ts

### OBSERVABLES FUNCTIONS ON EMBEDDED TIME SERIES ###

def distance_matrix(emb_ts: np.ndarray, m_norm = None, m = None):

    N = emb_ts.shape[1]

    dist_matrix = np.full((N,N), 0, dtype = np.float64)

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            dij = dist(x = emb_ts[:,i], y = emb_ts[:,j], m_norm = m_norm, m = m)

            dist_matrix[i,j] = dij
            dist_matrix[j,i] = dij

    return dist_matrix

# Correlation Sum from a Recurrence Plot
def corr_sum(dist_matrix: np.ndarray, r: float, w: int):

    N = dist_matrix.shape[0]

    if w == 0:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if dist_matrix[i,j] < r:

                    c += 1

        csum = (2/(N*(N-1)))*c

    else:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if dist_matrix[i,j] < r and (i - j) > w:

                    c += 1

        csum = (2/((N-w)*(N-w-1)))*c

    return csum

# Recurrence Plot for a single embeddend time series
def rec_plt(dist_matrix: np.ndarray, r: float, T: int, m_norm = None, m = None):

    N = dist_matrix.shape[0]

    rplt = np.full((T,T), 0, dtype = np.int8)

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            # Get value of theta
            if dist_matrix[i,j] < r:
                
                rplt[i,j] = 1
                rplt[j,i] = 1

    return rplt

# Spacetime Separation Plot for a single embeddend time series
def sep_plt(dist_matrix: np.ndarray, percentiles: np.array, T: int):

    N = dist_matrix.shape[0]

    n = percentiles.shape[0]

    splt = np.full((n,T), 0, dtype = np.float64)

    # Compose distribution of distances for each relative time distance
    for i in range(0,N):

        dist = []
        for j in range(0, N - i):

            dist.append(dist_matrix[j,i + j])

        if (N - i) > 2*n:

            perc = np.percentile(dist, percentiles)

            splt[:,i] = perc

    return splt

# Correlation Sum from a Recurrence Plot
def rec_corr_sum(recurrence_plot: np.ndarray, w: int):

    M = recurrence_plot.shape[0]

    # Get actual shape
    for i in range(0,M):

        if recurrence_plot[i,i] == 0:

            N = i

            break

    if w == 0:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                c += recurrence_plot[i,j]

        csum = (2/(N*(N-1)))*c

    else:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if(i - j) > w:

                    c += recurrence_plot[i,j]

        csum = (2/((N-w)*(N-w-1)))*c

    return csum

# Correlation Exponent computation
def corr_exp(log_csum: list, log_r: list, n_points: int, gauss_filter: bool, scale = None, cutoff = None):

    rlen = len(log_r) - n_points + 1

    ce =  []
    e_ce = []
    n_log_r = []
    for i in range(0,rlen):

        m = np.array([(log_csum[i+j+1] - log_csum[i+j])/(log_r[i+j+1] - log_r[i+j]) for j in range(0,n_points-1)])

        # Get value for error of slope
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = linregress(x = np.asarray(log_r)[i:i + n_points], y = log_csum[i:i + n_points], nan_policy = 'omit')

        ce.append(np.asarray(m).mean())
        e_ce.append(results.stderr)

        n_log_r.append(np.mean(np.asarray(log_r)[i:i + n_points]))

    ce = np.asarray(ce)
    e_ce = np.asarray(e_ce)

    n_log_r = np.asarray(n_log_r)

    # Apply Gaussian filtering for smoothing
    if gauss_filter == True:
        ce = gauss_kernel(function = ce, x = n_log_r, scale = scale, cutoff = cutoff, order = 0)
        e_ce = gauss_kernel(function = e_ce, x = n_log_r, scale = scale, cutoff = cutoff, order = 0)

    return  ce, e_ce

# Largest Lyapunov exponent for a single embedded time series [Rosenstein et al.]
def lyap(emb_ts: np.ndarray, lenght: int, m_period: int, sfreq: int, verbose = False):

    N = len(emb_ts)

    if N < m_period/10:
        print('Embedded data too short compared to average period')
        return

    ds = np.zeros((N,N), dtype = np.float64)

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i):
            
            dij = dist(emb_ts[i],emb_ts[j])
            ds[i,j] = dij
            ds[j,i] = dij

    # Construct separations trajectories with embedded data
    lnd = []
    for i, el in enumerate(ds):

        if i < N - lenght:

            # Select only distant points on the trajectory but not too far on the end
            jt = []
            for j in range(0,N):
                if abs(j-i) > m_period and j < N - lenght:
                    jt.append(j)

            if len(jt) != 0:
                # Get nearest neighbour
                d0 = np.min(el[jt])
                j = int(np.argwhere(el == d0))

            # Construct separation data
            for delta in range(0,lenght):
                if len(jt) != 0:
                    lnd.append(np.log(dist(emb_ts[i + delta],emb_ts[j + delta])))
                else:
                    lnd.append(np.nan)

    # Reshape results
    lnd = np.asarray(lnd).reshape((N - lenght ,lenght))

    # Get slope for largest Lyapunov exponent
    x = np.asarray([i for i in range (0,lenght)])

    with warnings.catch_warnings():
        if verbose == False:
            warnings.simplefilter('ignore')
            y = np.nanmean(lnd, axis = 0)*sfreq
        else:
            y = np.nanmean(lnd, axis = 0)*sfreq

    fit = linregress(x,y)

    lyap = fit.slope
    lyap_e = fit.stderr

    return lyap, lyap_e, x, y, fit

# Trial-wise function for correlation exponent computation
def rec_correlation_sum(trial_RP: list, log_r: list, w = None):

    # Initzialize results arrays
    CS = []
    E_CS = []

    abcd = trial_RP[0]
    abcd_ = trial_RP[1]
    for abc, abc_ in zip(abcd, abcd_):
        for ab, ab_ in zip(abc, abc_):
            for a, a_ in zip(ab, ab_):

                cs = corr_sum(recurrence_plot = a, w = w)

                CS.append(cs)
                E_CS.append(0)

    CS = np.asarray(CS)
    E_CS = np.asarray(E_CS)

    rshp = list(trial_RP[0].shape)[0:-2]

    CS = CS.reshape(rshp)
    E_CS = E_CS.reshape(rshp)

    return CS, E_CS

# Search for first plateau in Correlation Exponent
def ce_plateaus(trial_CE: np.ndarray, log_r: list, screen_points: int, resolution: int, backsteps: int, max_points: int):

    a = trial_CE[0]
    a_ = trial_CE[1]

    # Initzialize results array
    P = []
    Pe = []
    Pr = []
    for a1, a1_ in zip(a,a_):
        for a2, a2_ in zip(a1,a1_):
            
            c = 0
            for i, a_ in enumerate(a2):

                if np.isnan(a_) == False:

                    c += 1

                else:

                    c = 0

                if c == int(len(a2)/2):

                    nan_corr = i - int(len(a2)/2) + 1

                    break

            means = []
            for i in range(nan_corr, nan_corr + screen_points):

                m = np.mean(a2[i : i + resolution + 1])

                means.append(m)

            start = np.argmin(means) + nan_corr

            if start > backsteps:

                start = start - backsteps

            else:

                start = 0

            # Make longer fits
            fits = []
            finish = start + max_points

            sf_points = [i for i in range(start,finish)]

            intervals = []
            for f in sf_points:
                for s in range(start,f):
                    if abs(s - f) > resolution:
                        intervals.append([s,f])

            likelyhoods = []
            for point in intervals:

                ml = np.sum((a2[point[0]:point[1]] - np.nanmean(a2[point[0]:point[1]]))/(a2_[point[0]:point[1]]**2))/(point[1]-point[0])

                likelyhoods.append(ml)

            best_idx = np.argmin(likelyhoods)

            bounds = intervals[best_idx]

            mean = np.mean(a2[bounds[0]:bounds[1]])
            std = np.std(a2[bounds[0]:bounds[1]])

            rs = [bounds[0],bounds[1]]

            '''
            # Old method based on R-Value of fit :P VERY INEFFICIENT
            for point in intervals:

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')

                    fit = linregress(x = log_r[point[0]:point[1]], y = a2[point[0]:point[1]], nan_policy = 'omit')
                
                fits.append(fit)

            slope = [fit.slope for fit in fits]

            #rvalues = [abs(fit.rvalue) for fit in fits]
            
            t_idxs = []
            for i, r in enumerate(slope):

                if abs(r) < m_threshold:

                    t_idxs.append(i)
            
            if len(t_idxs) != 0:

                t_rvalues = [rvalues[i] for i in t_idxs]

                best_idx = t_idxs[np.argmin(t_rvalues)]

                bounds = intervals[best_idx]

                mean = np.mean(a2[bounds[0]:bounds[1]])
                std = np.std(a2[bounds[0]:bounds[1]])

                rs = [bounds[0],bounds[1]]
                

                # Initzialize weighted likelyhood list
                likelyhoods = []
                for i in t_idxs:

                    point = intervals[i]

                    ml = np.sum((a2[point[0]:point[1]] - np.mean(a2[point[0]:point[1]]))/(a2_[point[0]:point[1]]**2))

                    likelyhoods.append(ml)

                best_idx = t_idxs[np.argmin(likelyhoods)]

                bounds = intervals[best_idx]

                mean = np.mean(a2[bounds[0]:bounds[1]])
                std = np.std(a2[bounds[0]:bounds[1]])

                rs = [bounds[0],bounds[1]]
                
            else:

                mean = np.nan
                std = np.nan

                rs = [np.nan,np.nan]
            
            '''
            P.append(mean)
            Pe.append(std)
            Pr.append(rs)

    P = np.asarray(P).reshape(a.shape[:-1])
    Pe = np.asarray(Pe).reshape(a.shape[:-1])

    Pr = np.asarray(Pr).reshape([*a.shape[:-1],2])

    return P, Pe, Pr

# Find peaks in Correlation Exponent trial-wise results
def ce_peaks(trial_CE: np.ndarray, log_r: list, distance: int, height: list, prominence: list, width: list):

    a = trial_CE[0]
    a_ = trial_CE[1]

    # Initzialize results array
    P = []
    Pe = []
    Pr = []
    for a1, a1_ in zip(a,a_):
        for a2, a2_ in zip(a1,a1_):

            # Search for the last peak
            peaks = find_peaks(a2, distance = distance, height = height, prominence = prominence, width = width)

            if len(peaks[0]) != 0:
                P.append(a2[peaks[0][-1]])
                Pe.append(a2_[peaks[0][-1]])
                Pr.append(log_r[peaks[0][-1]])
            else:
                P.append(np.nan)
                Pe.append(np.nan)
                Pr.append(np.nan)

    P = np.asarray(P).reshape(a.shape[:-1])
    Pe = np.asarray(Pe).reshape(a.shape[:-1])
    #P = np.concatenate((P[np.newaxis], Pe[np.newaxis]), axis = 0)

    Pr = np.asarray(Pr).reshape(a.shape[:-1])

    return P, Pe, Pr

### SUB-TRIAL WISE FUNCTIONS FOR OBSERVABLES COMPUTATION ###

# Get epoched 
def epochs(evoked: mne.Evoked, s_evoked: mne.Evoked, ch_list: list|tuple, fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]
    
    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)
    s_evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    # Initzialize result array
    EP = []
    E_EP = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:

        TS = []
        E_TS = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            e_ts = s_evoked.get_data(picks = cl)

            ts = ts.mean(axis = 0)
            e_ts = e_ts.mean(axis = 0)
            
            TS.append(ts)
            E_TS.append(e_ts)

        # Loop around pois time series
        for ts, e_ts in zip(TS, E_TS):

            EP.append(ts)
            E_EP.append(e_ts)

    else:

        TS = evoked.get_data(picks = ch_list)
        E_TS = s_evoked.get_data(picks = ch_list)

        # Loop around pois time series
        for ts, e_ts in zip(TS, E_TS):

            EP.append(ts)
            E_EP.append(e_ts)

    return EP, E_EP

# Get epochs frequency spectrum
def spectrum(evoked: mne.Evoked, s_evoked: mne.Evoked, ch_list: list|tuple, N: int, wf: float, fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]
    
    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)
    s_evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    # Initzialize result array
    SP = []
    E_SP = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:

        TS = []
        E_TS = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            e_ts = s_evoked.get_data(picks = cl)

            n = evoked.nave

            ts = ts.mean(axis = 0)
            e_ts = e_ts.mean(axis = 0)
            
            TS.append(ts)
            E_TS.append(e_ts)

        # Loop around pois time series
        for ts, e_ts in zip(FT, E_FT):

            psd, _ = mne.time_frequency.psd_array_welch(ts,
            sfreq = evoked.info['sfreq'],  # Sampling frequency from the evoked data
            fmin = evoked.info['highpass'], fmax = evoked.info['lowpass'],  # Focus on the filter range
            n_fft = int(len(ts)*wf),
            n_per_seg = int(len(ts)/wf),  # Length of FFT (controls frequency resolution)
            verbose = False)
            
            # Generate time series on a gaussian noise hypotheses if we have more than one trial
            if n != 1:
                
                psd_ = []
                for i in range(0,N):

                    ts_r = np.random.normal(loc = ts, scale = e_ts*np.sqrt(n))

                    psd_r, _ = mne.time_frequency.psd_array_welch(ts_r,
                    sfreq = evoked.info['sfreq'],  # Sampling frequency from the evoked data
                    fmin = evoked.info['highpass'], fmax = evoked.info['lowpass'],  # Focus on the filter range
                    n_fft = int(len(ts)*wf),
                    n_per_seg = int(len(ts)/wf),  # Length of FFT (controls frequency resolution)
                    verbose = False)

                    psd_.append(psd_r)

                e_psd = np.asarray(psd_).std(axis = 0)/np.sqrt(N)

            # Otherwise compute fft without any error output
            else:

                e_psd = np.zeros(len(psd))

            SP.append(psd)
            E_SP.append(e_psd)

    else:

        TS = evoked.get_data(picks = ch_list)
        E_TS = s_evoked.get_data(picks = ch_list)

        n = evoked.nave

        # Loop around pois time series
        for ts, e_ts in zip(TS, E_TS):

            psd, _ = mne.time_frequency.psd_array_welch(ts,
            sfreq = evoked.info['sfreq'],  # Sampling frequency from the evoked data
            fmin = evoked.info['highpass'], fmax = evoked.info['lowpass'],  # Focus on the filter range
            n_fft = int(len(ts)*wf),
            n_per_seg = int(len(ts)/wf),  # Length of FFT (controls frequency resolution)
            verbose = False)

            # Generate time series on a gaussian noise hypotheses if we have more than one trial
            if n != 1:
                
                psd_ = []
                for i in range(0,N):

                    ts_r = np.random.normal(loc = ts, scale = e_ts*np.sqrt(n))

                    psd_r, _ = mne.time_frequency.psd_array_welch(ts_r,
                    sfreq = evoked.info['sfreq'],  # Sampling frequency from the evoked data
                    fmin = evoked.info['highpass'], fmax = evoked.info['lowpass'],  # Focus on the filter range
                    n_fft = int(len(ts)*wf),
                    n_per_seg = int(len(ts)/wf),  # Length of FFT (controls frequency resolution) # Length of FFT (controls frequency resolution)
                    verbose = False)

                    psd_.append(psd_r)

                e_psd = np.asarray(psd_).std(axis = 0)/np.sqrt(N)

            # Otherwise compute fft without any error output
            else:

                e_psd = np.zeros(len(psd))

            SP.append(psd)
            E_SP.append(e_psd)

    return SP, E_SP

# Get epoched 
def persistence(evoked: mne.Evoked, s_evoked: mne.Evoked, ch_list: list|tuple, fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]
    
    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)
    s_evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    # Initzialize result array
    PS = []
    E_PS = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:

        TS = []
        E_TS = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            e_ts = s_evoked.get_data(picks = cl)

            ts = ts.mean(axis = 0)
            e_ts = e_ts.mean(axis = 0)
            
            TS.append(ts)
            E_TS.append(e_ts)

        # Loop around pois time series
        for ts, e_ts in zip(TS, E_TS):

            PS.append(ts)
            E_PS.append(e_ts)

    else:

        TS = evoked.get_data(picks = ch_list)
        E_TS = s_evoked.get_data(picks = ch_list)

        # Loop around pois time series
        for ts, e_ts in zip(TS, E_TS):

            PS.append(ts)
            E_PS.append(e_ts)

    return EP, E_EP

# Delay time of channel time series of a specific trial
def delay_time(evoked: mne.Evoked, ch_list: list|tuple,
               method: str, clst_method: str, fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]
    
    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    # Initzialize result array
    tau = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:
        

        TS = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            #ts = ts.mean(axis = 0)
            
            TS.append(ts)
        
        for ts in TS:

            ctau = []

            if method == 'mutual_information':

                [ctau.append(MI_for_delay(t)) for t in ts]

            elif method == 'autocorrelation':

                [ctau.append(autoCorrelation_tau(t)) for t in ts]

            if clst_method == 'avg':

                tau.append(np.asarray(ctau).mean())

            elif cmst_methd == 'max':

                tau.append(np.asarray(ctau).max())

    else:

        TS = evoked.get_data(picks = ch_list)
    
        # Loop around pois time series
        for ts in TS:

            if method == 'mutual_information':

                tau.append(MI_for_delay(ts))

            elif method == 'autocorrelation':

                tau.append(autoCorrelation_tau(ts))

    # Returns list in -C style ordering

    return tau

# Correlation sum of channel time series of a specific trial [NOW IT USES RECURRENCE PLOTS]
def correlation_sum(evoked: mne.Evoked, ch_list: list|tuple,
                    embeddings: list, tau: str|int, w: int, rvals: list, 
                    m_norm: bool, fraction = [0,1], cython = False):

    if cython == True:

        from cython_modules import c_core

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]

    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    # Initzialize result array
    CS = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:
        

        TS = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            #ts = ts.mean(axis = 0)
            
            TS.append(ts)
        
        for ts in TS:

            # Compile delay time list for each component
            tau_ = []
            for t in ts:
                
                if tau == 'mutual_information':

                    tau_.append(MI_for_delay(t))

                elif tau == 'autocorrelation':

                    tau_.append(autoCorrelation_tau(t))

                elif type(tau) == int:

                    tau_.append(tau)

            for m in embeddings:
                
                emb_ts = []
                emb_lenghts = []
                for i, t in enumerate(ts):

                    emb_t = td_embedding(t, embedding = m, tau = tau_[i])

                    l = len(emb_t)

                    emb_ts.append(emb_t)
                    emb_lenghts.append(l)

                # Set embedded time series to same lenght for array transformation
                emb_ts = [emb_t[0:np.asarray(emb_lenghts).min()] for emb_t in emb_ts]

                emb_ts = np.asarray(emb_ts)

                emb_ts = np.swapaxes(emb_ts, 1, 2)
                emb_ts = np.swapaxes(emb_ts, 0, 1)

                emb_ts = np.asarray([emb_ts[i,j] for j in range(0,emb_ts.shape[1]) for i in range(0,emb_ts.shape[0])], dtype = np.float64)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                for r in rvals:

                    if cython == False:

                        cs = corr_sum(dist_matrix = dist_matrix, r = r, w = w)

                    else:

                        cs = c_core.corr_sum(dist_matrix = dist_matrix, r = r, w = w)

                    CS.append(cs)

    else:

        TS = evoked.get_data(picks = ch_list)
    
        # Loop around pois time series
        for ts in TS:
                
            if tau == 'mutual_information':

                tau_ = MI_for_delay(ts)

            elif tau == 'autocorrelation':

                tau_ = autoCorrelation_tau(ts)

            elif type(tau) == int:

                tau_ = tau
            
            for m in embeddings:
                emb_ts = td_embedding(ts, embedding = m, tau = tau_)

                emb_ts = np.asarray(emb_ts)

                emb_ts = np.swapaxes(emb_ts, 0, 1)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                for r in rvals:

                    if cython == False:

                        cs = corr_sum(dist_matrix = dist_matrix, r = r, w = w)

                    else:

                        cs = c_core.corr_sum(dist_matrix = dist_matrix, r = r, w = w)

                    CS.append(cs)

    # Returns list in -C style ordering

    return CS

# Recurrence Plot of channel time series of a specific trial
def recurrence_plot(evoked: mne.Evoked, ch_list: list|tuple,
                    embeddings: list, tau: str|int, rvals: list, 
                    m_norm: bool, fraction = [0,1], cython = False):

    if cython == True:

        from cython_modules import c_core

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]

    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    T = len(evoked.times)

    # Initzialize result array
    RP = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:

        TS = []
        for cl in ch_list:

            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            #ts = ts.mean(axis = 0)
            
            TS.append(ts)
        
        for ts in TS:

            # Compile delay time list for each component
            tau_ = []
            for t in ts:

                if tau == 'mutual_information':

                    tau_.append(MI_for_delay(t))

                elif tau == 'autocorrelation':

                    tau_.append(autoCorrelation_tau(t))

                elif type(tau) == int:

                    tau_.append(tau)

            for m in embeddings:

                emb_ts = []
                emb_lenghts = []
                for i, t in enumerate(ts):

                    emb_t = td_embedding(t, embedding = m, tau = tau_[i])

                    l = len(emb_t)

                    emb_ts.append(emb_t)
                    emb_lenghts.append(l)

                # Set embedded time series to same lenght for array transformation
                emb_ts = [emb_t[0:np.asarray(emb_lenghts).min()] for emb_t in emb_ts]

                emb_ts = np.asarray(emb_ts)

                emb_ts = np.swapaxes(emb_ts, 1, 2)
                emb_ts = np.swapaxes(emb_ts, 0, 1)

                emb_ts = np.asarray([emb_ts[i,j] for j in range(0,emb_ts.shape[1]) for i in range(0,emb_ts.shape[0])], dtype = np.float64)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                for r in rvals:

                    if cython == False:

                        rp = rec_plt(dist_matrix = dist_matrix, r = r, T = T)

                    else:

                        rp = c_core.rec_plt(dist_matrix = dist_matrix, r = r, T = T)

                    RP.append(rp)

    else:

        TS = evoked.get_data(picks = ch_list)
    
        # Loop around pois time series
        for ts in TS:
                
            if tau == 'mutual_information':

                tau_ = MI_for_delay(ts)

            elif tau == 'autocorrelation':

                tau_ = autoCorrelation_tau(ts)

            elif type(tau) == int:

                tau_ = tau
            
            for m in embeddings:

                emb_ts = td_embedding(ts, embedding = m, tau = tau_)

                emb_ts = np.asarray(emb_ts)

                emb_ts = np.swapaxes(emb_ts, 0, 1)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                for r in rvals:

                    if cython == False:

                        rp = rec_plt(emb_ts = emb_ts, r = r, T = T)

                    else:

                        rp = c_core.rec_plt(emb_ts = emb_ts, r = r, T = T)

                    RP.append(rp)

    # Returns list in -C style ordering

    return RP

# Correlation sum of channel time series of a specific trial [NOW IT USES RECURRENCE PLOTS]
def separation_plot(evoked: mne.Evoked, ch_list: list|tuple,
                    embeddings: list, tau: str|int, percentiles: list,
                    m_norm: bool, fraction = [0,1], cython = False):

    if cython == True:

        from cython_modules import c_core

    # Apply fraction to time series
    times = evoked.times

    start = int(fraction[0]*len(times))
    finish = int(fraction[1]*len(times)) - 1

    # Trim time series according to fraction variable
    tmin = times[start]
    tmax = times[finish]

    evoked.crop(tmin = tmin, tmax = tmax, include_tmax = False)

    T = len(evoked.times)

    percentiles = np.asarray(percentiles, dtype = np.int8)

    # Initzialize result array
    SP = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:
        

        TS = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            #ts = ts.mean(axis = 0)
            
            TS.append(ts)
        
        for ts in TS:

            # Compile delay time list for each component
            tau_ = []
            for t in ts:
                
                if tau == 'mutual_information':

                    tau_.append(MI_for_delay(t))

                elif tau == 'autocorrelation':

                    tau_.append(autoCorrelation_tau(t))

                elif type(tau) == int:

                    tau_.append(tau)

            for m in embeddings:
                
                emb_ts = []
                emb_lenghts = []
                for i, t in enumerate(ts):

                    emb_t = td_embedding(t, embedding = m, tau = tau_[i])

                    l = len(emb_t)

                    emb_ts.append(emb_t)
                    emb_lenghts.append(l)

                # Set embedded time series to same lenght for array transformation
                emb_ts = [emb_t[0:np.asarray(emb_lenghts).min()] for emb_t in emb_ts]

                emb_ts = np.asarray(emb_ts)

                emb_ts = np.swapaxes(emb_ts, 1, 2)
                emb_ts = np.swapaxes(emb_ts, 0, 1)

                emb_ts = np.asarray([emb_ts[i,j] for j in range(0,emb_ts.shape[1]) for i in range(0,emb_ts.shape[0])], dtype = np.float64)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                    sp = sep_plt(dist_matrix = dist_matrix, percentiles = percentiles, T = T)

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                    sp = c_core.sep_plt(dist_matrix = dist_matrix, percentiles = percentiles, T = T)

                SP.append(sp)

    else:

        TS = evoked.get_data(picks = ch_list)
    
        # Loop around pois time series
        for ts in TS:
                
            if tau == 'mutual_information':

                tau_ = MI_for_delay(ts)

            elif tau == 'autocorrelation':

                tau_ = autoCorrelation_tau(ts)

            elif type(tau) == int:

                tau_ = tau
            
            for m in embeddings:
                emb_ts = td_embedding(ts, embedding = m, tau = tau_)

                emb_ts = np.asarray(emb_ts)

                emb_ts = np.swapaxes(emb_ts, 0, 1)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                    sp = sep_plt(dist_matrix = dist_matrix, percentiles = percentiles, T = T)

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                    sp = c_core.sep_plt(dist_matrix = dist_matrix, percentiles = percentiles, T = T)

                SP.append(sp)

    # Returns list in -C style ordering

    return SP

# Information dimension of channel time series of a specific trial
def information_dimension(evoked: mne.Evoked, ch_list: list|tuple,
                          embeddings: list, tau: int, 
                          fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    # Trim time series according to fraction variable
    tmin = times[int(fraction[0]*(len(times)-1))]
    tmax = times[int(fraction[1]*(len(times)-1))]

    evoked.crop(tmin = tmin, tmax = tmax)

    # Initzialize result array
    D2 = []
    e_D2 = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:
        

        tl = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            ts = ts.mean(axis = 0)
            
            tl.append(ts)
        
        TS = np.asarray(tl)

    else:    

        TS = evoked.get_data(picks = ch_list)
    
    # Loop around pois time series
    for ts in TS:
        
        for m in embeddings:
            emb_ts = td_embedding(ts, embedding = m, tau = tau)

            d2, e_d2 = idim(emb_ts, m_period = tau)

            D2.append(d2)
            e_D2.append(e_d2)

    return D2, e_D2

# Largest lyapunov exponent of channel time series
def lyapunov(evoked: mne.Evoked, ch_list: list | tuple, 
             embeddings: list, tau: int, lenght: int, avT = None,
             fraction = [0,1], verbose = False):

    # Get sampling frequency
    sfreq = evoked.info['sfreq']

    # Apply fraction to time series
    times = evoked.times

    # Trim time series according to fraction variable
    tmin = times[int(fraction[0]*(len(times)-1))]
    tmax = times[int(fraction[1]*(len(times)-1))]

    evoked.crop(tmin = tmin, tmax = tmax)

    # Initzialize result arrays
    ly = []
    # Error given by alghorithm, to use in case 'avg_trials == True'
    ly_e = []

    # Check if we are clustering electrodes
    if type(ch_list) == tuple:

        tl = []
        for cl in ch_list:
            
            # Get average time series of the cluster
            ts = evoked.get_data(picks = cl)
            ts = ts.mean(axis = 0)
            
            tl.append(ts)
        
        TS = np.asarray(tl)

    else:    

        TS = evoked.get_data(picks = ch_list)
    
    # Loop around pois tim series
    for ts in TS:

        # Get mean period of the time series through power spectrum analysis
        # this method is not very robust to noise if the estimated period is too
        # large for the embeddin dimension
        if avT == None:
            f, ps = periodogram(ts)
            avT = np.sum(ps)/np.sum(f*ps)
            if verbose == True:
                print('Average period: ',avT)
        
        for m in embeddings:
            emb_ts = td_embedding(ts, embedding = m, tau = tau)
            
            l, l_e, x, y, fit = lyap(emb_ts, m_period = avT, lenght = lenght, sfreq = sfreq, verbose = verbose)

            '''
            # Plotting for correct estimation of optimal lenght,
            # the system is bounded so is the separation of trajectories

            plt.plot(x,y)
            plt.plot(x, (fit.slope*x + fit.intercept))

            plt.show()
            plt.close()
            '''

            ly.append(l)
            ly_e.append(l_e)

    return ly, ly_e

# Sub-wise function for correlation exponent computation
def correlation_exponent(sub_log_CS: list, n_points: int, log_r: list,
                         gauss_filter: None, scale: None, cutoff = None):

    # Reduced rvals lenght for mobile average
    rlen = len(log_r) - n_points + 1

    # Initzialize results arrays
    CE = []
    E_CE = []

    abc = sub_log_CS[0]
    abc_ = sub_log_CS[1]
    for ab, ab_ in zip(abc, abc_):
        for a, a_ in zip(ab, ab_):

            ce, e_ce = corr_exp(log_csum = a, log_r = log_r, n_points = n_points, gauss_filter = gauss_filter, scale = scale, cutoff = cutoff)

            CE.append(ce)
            E_CE.append(e_ce)

    CE = np.asarray(CE)
    E_CE = np.asarray(E_CE)

    rshp = list(sub_log_CS[0].shape)

    rshp[-1] = rlen

    CE = CE.reshape(rshp)
    E_CE = E_CE.reshape(rshp)

    return CE, E_CE

### RESULTS MANIPULATION FUNCTION ###

# Select some electrodes from global results
def reduceCS(exp_name: str, avg_trials: bool, ch_list: list, average = False, clust_lb = 'G', nlabel = None):

    path = obs_path(exp_name = exp_name, clust_lb = clust_lb, obs_name = 'corrsum', avg_trials = avg_trials)

    CS = np.load(path + 'corrsum.npy')

    with open(path + 'variables.json', 'r') as f:
        d = json.load(f)

    log_r = d['log_r']

    # Check if we are loading a clusterd calculation [certaingly not suitable]
    if d['clustered'] == True:
        print('Clustered results are not valid for array reduction')
        return

    ch_idx = name_toidx(ch_list, exp_name = exp_name)

    shp = np.asarray(CS.shape)
    shp[3] = 0

    CSred = np.empty(shp)
    for idx in ch_idx:
        CSred = np.concatenate((CSred, CS[:,:,:,idx,:,:][:,:,:,np.newaxis,:,:]), axis = 3)

    # Average results for plotting
    if average == True:
        CSred = CSred.mean(axis = 3)
        CSred = CSred[:,:,:,np.newaxis,:,:]

    # Save results in a new directory
    if nlabel != None:

        # Change dictionary entry for save    
        d['pois'] = ch_list

        sv_path = obs_path(exp_name = exp_name, clust_lb = nlabel, obs_name = 'corrsum', avg_trials = avg_trials)

        os.makedirs(sv_path, exist_ok = True)
        np.save(sv_path + 'corrsum.npy', CSred)

        # Save new variables dictionary
        with open(sv_path + 'variables.json', 'w') as f:
            json.dump(d, f)

    return CSred, log_r

#   Following functions do not implement mne data structure and each of them averages over trials
#   They are useful because they can average over cluster of electrodes as well but this doesn't seem
#   to be very easy to implement for mapping in mne.

#   They are fast and can be useful later because some observables need more averaging


# Prepare dataframe to store results with the following hierarchy:

#   [conditions[ electrodes | groupsofelectrode [ metrics ]]]

#   !!!THIS HIERARCHY HAS TO BE FOLLOWED IN EACH FUNCTION FOR CONSISTENCY!!!

def init_Frame(conditions: list, channels_idx: list | tuple, metrics: list):

    df_cols = ['SubID']

    for cnd in conditions:
        
        if type(channels_idx) == tuple:

            for chs in channels_idx:

                idx_val = []
                for st in chs:
                    idx_val = idx_val + [int(st)]
                idx_str = str(idx_val)

                for met in metrics:

                    df_cols = df_cols + [met + ' ' + cnd + ' ' + idx_str]

        elif type(channels_idx) == list:
            
            idx_val = []
            for chs in channels_idx:

                idx_val = idx_val + [int(chs)]
            
            idx_str = str(idx_val)

            for met in metrics:

                df_cols = df_cols + [met + ' ' + cnd + ' ' + idx_str]

        else:
                
            print('Channel indexes format error, check channel_idx data type:\n' + str(type(chs)))
            print('Functions that take it could break!')

    df = pd.DataFrame(columns = df_cols)

    return df


# Process time series for autocorrelation time and minimum embedding dimension

def SSub_ntau(subID: str, conditions: list, channels_idx: list | tuple):
    
    results = [subID]

    folder = bw_path + 'subj' + subID + '_band_resample/'
    all_files = os.listdir(folder)

    for cond in conditions:
        
        my_cond_files = [f for f in all_files if cond in f ]

        # Check for the right idxs format
        if type(channels_idx) == tuple:
            
            untup = []
            for chs in channels_idx:
                
                untup = untup + chs

            all_trials = np.empty((0,len(untup),Tst))
                
            for f in my_cond_files:

                path = folder+f
                    
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup]

                
                all_trials = np.concatenate((all_trials, data[np.newaxis,:]), axis = 0)
            
            #Average across trials
            avg_trials = all_trials.mean(axis = 0)

            l0 = 0
            for chs in channels_idx:

                raw_ts = np.empty((0,Tst))

                #Let's search the minimum embedding dimension (Over trial average)
                #Get the time series data (Average across trials)
                raw_ts = np.concatenate((raw_ts, avg_trials[l0:l0 + len(chs),:].mean(axis=0)[np.newaxis,:]))
                raw_ts = np.reshape(raw_ts,Tst)

                l0 += len(chs)

                #Search autocorrelation time (Autocorelation decay, only for linear system)
                #tau_adim = autoCorrelation_tau(raw_ts)
              
                #Search autocorrelation time (Mutal Information Minimum)
                tau_adim = MI_for_delay(raw_ts, k = k_set)

                #Search minimum embedding dimension (False Nearest Neighbor)
                perc_FNN, n = FNN_n(raw_ts, tau_adim, Rtol = Rtol_set)

                results = results + [tau_adim, n]

        elif type(channels_idx) == list:

            all_trials = np.empty((0,Tst))

            for f in my_cond_files:

                path = folder+f

                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][channels_idx]
                all_trials = np.concatenate((all_trials, data.mean(axis=0)[np.newaxis, :]))

            #Let's search the minimum embedding dimension (Over trial average)
            #Get the time series data (Average across trials)
            raw_ts = all_trials.mean(axis=0)
            
            #Search autocorrelation time (Autocorelation decay, only for linear system)
            #tau_adim = autoCorrelation_tau(raw_ts)
        
            #Search autocorrelation time (Mutal Information Minimum)
            tau_adim = MI_for_delay(raw_ts, k = k_set)

            #Search minimum embedding dimension (False Nearest Neighbor)
            perc_FNN, n = FNN_n(raw_ts, tau_adim, Rtol = Rtol_set)

            results = results + [tau_adim, n]
        
        else:

            print('Channel indexes format error, check channel_idx data type:\n' + str(type(channels_idx)))
            return   
            
    print('Sub' + subID + ': Done! ')

    #Returns list of results following the hierarchy
    return results


# Embed the time series with a fixed tau in a 2-dimensional plot, for easy pics

# This is very stupid and hardly scalable, but they do their job
def twodim_graphs(subID: str, tau: int, trim: int, conditions: list, channels_idx: list | tuple):
    
    results = [subID]

    folder = bw_path + 'subj' + subID + '_band_resample/'
    all_files = os.listdir(folder)
    
    save_path = bw_pics_path + '/2dim_emb/'+ str(conditions) + subID + '/'
    os.makedirs(save_path, exist_ok = True)

    if type(channels_idx) == tuple:

        raw_ts = np.empty((0, len(channels_idx), Tst))

    elif type(channels_idx) == list:

        raw_ts = np.empty((0, Tst))
        
    else:

        print('Channel indexes format error, check channel_idx data type:\n' + str(type(channels_idx)))
        return   

    for c_cond, cond in enumerate(conditions):
        
        my_cond_files = [f for f in all_files if cond in f ]

        # Check for the right idxs format
        if type(channels_idx) == tuple:
            
            untup = []
            for chs in channels_idx:
                
                untup = untup + chs

            all_trials = np.empty((0,len(untup),Tst))

            for f in my_cond_files:

                path = folder+f
                    
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup]
                
                all_trials = np.concatenate((all_trials, data[np.newaxis,:]), axis = 0)

            #Average across trials
            avg_trials = all_trials.mean(axis = 0)
            raw_chs_ts = np.empty((0, Tst))
            
            #emb_memory = np.empty((2, len(conditions),len(channels_idx),Tst - 2*trim - tau))

            l0 = 0
            for chs in channels_idx:
                
                raw_chs_ts = np.concatenate((raw_chs_ts, avg_trials[l0:l0 + len(chs),:].mean(axis=0)[np.newaxis,:]), axis = 0)
                l0 += len(chs)
            
            raw_ts = np.concatenate((raw_ts, raw_chs_ts[np.newaxis, :,]), axis = 0)

        elif type(channels_idx) == list:

            all_trials = np.empty((0,Tst))

            for f in my_cond_files:

                path = folder+f

                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][channels_idx]
                all_trials = np.concatenate((all_trials, data.mean(axis=0)[np.newaxis, :]))

            raw_ts = np.concatenate((raw_ts, all_trials.mean(axis = 0)[np.newaxis, :]), axis = 0)

    #Printing area

    #Initzialize embedding coordinates
    emb = np.empty((2, Tst - 2*trim - tau))

    if type(channels_idx) == tuple:

        for c_chs, chs in enumerate(channels_idx):

            idx_val = []
            for st in chs:
                idx_val = idx_val + [int(st)]
            idx_str = str(idx_val)
            
            fig, ax = plt.subplots()

            for c_cond, cond in enumerate(conditions):
                
                emb[0,:] = raw_ts[c_cond, c_chs, trim : Tst - trim - tau]
                emb[1,:] = raw_ts[c_cond, c_chs, trim + tau : Tst - trim]

                ax.plot(emb[0,:], emb[1,:], 'o-', linewidth = 1, markersize = 3, label = cond)

            ax.set(title = str(conditions) + idx_str)

            plt.savefig(save_path + idx_str + '.png')
            plt.close()

    if type(channels_idx) == list:

        idx_val = []
        for chs in channels_idx:

            idx_val = idx_val + [int(chs)]

        idx_str = str(idx_val)
        
        fig, ax = plt.subplots()

        for c_cond, cond in enumerate(conditions):

            emb[0,:] = raw_ts[c_cond, trim : Tst - trim - tau]
            emb[1,:] = raw_ts[c_cond, trim + tau : Tst - trim]

            ax.plot(emb[0,:], emb[1,:], 'o-', linewidth = 1, markersize = 3, label = cond)

        ax.set(title = str(conditions) + idx_str)

        plt.savefig(save_path + idx_str + '.png')
        plt.close()

    print('Sub' + subID + ': Done! ')

    return

# Core structure for navigating the raw dataset (not in MNE data structure)
def core(subID: str, conditions: list, channels_idx: list | tuple):
    
    results = [subID]

    folder = bw_path + 'subj' + subID + '_band_resample/'
    all_files = os.listdir(folder)
    # cond = 'S_1' # unconscious

    # Loop over conditi
    for cond in conditions:
        
        my_cond_files = [f for f in all_files if cond in f ]

        # Check for the right idxs format
        if type(channels_idx) == tuple:
            
            untup = []
            for chs in channels_idx:
                
                untup = untup + chs

            all_trials = np.empty((0,len(untup),Tst))
                
            for f in my_cond_files:

                path = folder+f
                    
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup]

                
                all_trials = np.concatenate((all_trials, data[np.newaxis,:]), axis = 0)
            
            #Average across trials
            avg_trials = all_trials.mean(axis = 0)

            l0 = 0
            for chs in channels_idx:

                raw_ts = avg_trials[l0:l0 + len(chs),:].mean(axis=0)

                l0 += len(chs)

                '''

                Do whatever computation you want 
                on the time series analysis HERE!
            
                '''

        elif type(channels_idx) == list:

            all_trials = np.empty((0,Tst))

            for f in my_cond_files:

                path = folder+f

                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][channels_idx]
                all_trials = np.concatenate((all_trials, data.mean(axis=0)[np.newaxis, :]))

            #Get the time series data (Average across trials)
            raw_ts = all_trials.mean(axis=0)
            
            '''

            Do whatever computation you want 
            on the time series analysis HERE!
            
            '''

        else:

            print('Channel indexes format error, check channel_idx data type:\n' + str(type(channels_idx)))
            return   
            
    print('Sub' + subID + ': Done! ')

    #Remebrer the hierarchy! If it applies
    return results

# OLD FUNCTIONS I DON'T NEED NOW

# Information Dimension for a single embedded time series with 2NN-estimation [Krakovská-Chvosteková]
def idim_old(emb_ts: np.ndarray, m_period: int):

    N = len(emb_ts)

    #Initzialize distances array
    ds = np.zeros((N,N), dtype = np.float64)
    for i in range(0,N):
        for j in range(0,N):
            dij = dist(emb_ts[i],emb_ts[j])

            ds[i,j] = dij
            ds[j,i] = dij

    #Initzialize reults array
    ls = []
    for i in range(0,N):
        di = ds[i]
        
        sorted_di_idx = np.argsort(di)

        # Store first two non-zero smallest distances (far away on the trajectory)
        r_idx = [0,0]
        c = 0
        m = i
        for d in sorted_di_idx:
            if di[d] != 0 and abs(i - d) > m_period and abs(m - d) > m_period:
                r_idx[c] = d
                m = d
                c += 1
            if c == 2:
                break

        r1 = di[r_idx[0]]
        r2 = di[r_idx[1]]

        r12 = ds[r_idx[0],r_idx[1]]

        # Get one of three possible values for the estimator
        if r12 <= r1:
            l = np.log(3/2)/np.log(r2/r1)
        elif r12 > r1 and r12 <= r2:
            l = np.log(3)/np.log(r2/r1)
        elif r12 > r2:
            l = np.log(2)/np.log(r2/r1)

        ls.append(l)

    idim = np.median(np.asarray(ls))
    e_idim = np.std(np.asarray(ls))

    return idim, e_idim
