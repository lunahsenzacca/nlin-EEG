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

    if obs_name == 'delay':

        M = np.load(obs_path + 'delay.npz')
        x = variables['pois']

        X = [x]

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
            if exp_name == 'bmasking':
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup][np.newaxis]

            elif exp_name == 'bmasking_dense':
                mne_data = mne.read_epochs(path, preload = True, verbose = False)

                data = mne_data.get_data(picks = untup)

            all_trials = np.concatenate((all_trials, data), axis = 0)

        data_list.append(all_trials)

    return data_list

# Create info file for specific datased
def toinfo(exp_name: str, ch_type = 'eeg'):

    # Get electrodes labels
    ch_list = maind[exp_name]['pois']

    ch_types = [ch_type for n in range(0, len(ch_list))]

    # Get sampling frequency
    freq = maind[exp_name]['f']

    info = mne.create_info(ch_list, ch_types = ch_types, sfreq = freq)
    #info.set_montage(maind[exp_name]['montage'])

    return info

# Function for subwise evoked conversion from list of arrays
def list_toevoked(data_list: list, subID: str, exp_name: str, avg_trials: bool, z_score: bool, alt_sv: str):
    
    # There are different number of trials for each condition,
    # so a simple ndarray is inconvenient. We use a list of ndarray instead

    # The list index cycles faster around the conditions and slower around the subject

    # Create info file
    info = toinfo(exp_name = exp_name)

    conditions = list(maind[exp_name]['conditions'].values())

    # Initialize evokeds list
    evokeds = []

    # Cycle around conditions
    for i, array in enumerate(data_list):

        # 'array' structure
        # Axis 0 = Trials
        # Axis 1 = Electrodes
        # Axis 2 = Time points

        # Average across same condition trials
        if avg_trials == True:

            n_trials = array.shape[0]
            avg = array.mean(axis = 0)

            # Apply Z-Score
            if z_score == True:

                avg = zscore(avg[np.newaxis])
                avg = avg[0]

            ev = mne.EvokedArray(avg, info, nave = n_trials, comment = conditions[i])
            evokeds.append(ev)

        # Keep each individual trial
        else:

            # Apply zscore normalization
            if z_score == True:

                array = zscore(array)

            for trl in array:

                ev = mne.EvokedArray(trl, info, nave = 1, comment = conditions[i])
                evokeds.append(ev)

    # This is meant for testing in notebooks
    if alt_sv != None:

        # Create directory
        os.makedirs(alt_sv, exist_ok = True)

        # Evoked file directory
        sv_path = alt_sv + subID + '-ave.fif'

        mne.write_evokeds(sv_path, evokeds, overwrite = True, verbose = False)
 
    return evokeds

# Create evoked file straight from raw data
def toevoked(subID: str, exp_name: str, avg_trials: bool, z_score = False, alt_sv = None):

    # Create data list
    data_list = raw_tolist(subID = subID, exp_name = exp_name)

    # Generate evokeds
    evokeds = list_toevoked(data_list = data_list, subID = subID, exp_name = exp_name, avg_trials = avg_trials, z_score = z_score, alt_sv = alt_sv)

    return evokeds

# Load evoked files of a specific subject
def loadevokeds(exp_name: str, avg_trials: bool, subID: str, conditions: list):

    # Select correct path for data
    if avg_trials == True:
        file_p = maind[exp_name]['directories']['avg_data']
    else:
        file_p = maind[exp_name]['directories']['trl_data']

    evokeds = mne.read_evokeds(file_p + subID + '-ave.fif', verbose = False)

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
def loadresults(obs_path: str, obs_name: str):

    M, X, variables = obs_data(obs_path = obs_path, obs_name = obs_name)

    len_s = len(variables['subjects'])
    len_c = len(variables['conditions'])

    results = []
    for s in range(0, len_s):

        c_results = []
        for c in range(0, len_c):

            c_results.append(M[M.files[len_c*s + c]])

        results.append(c_results)

    return results, X, variables

# Create one dimensional list of results per subject per condition
def flat_results(results: list):

    # Flatten results nested list
    flat = [x for xss in results for xs in xss for x in xs]

    # Save separation coordinates for 'collapse_trials' functions
    points = [[len(results[i][j]) for j in range(0,len(results[i]))] for i in range(0,len(results))]

    return flat, points

# Function for trials averaging and homogeneous array generation (works for already averaged trials as well)
def collapse_trials(results: list, points: list, fshape: list, e_results = None):

    if e_results == None:
        print('No observable error in input, zeros will be appended in results along trial error')

    # Initzialize list of homegenous arrays
    RES = []

    if e_results != None:
        OBS_STD = []

    # Make homogeneous arrays for each subject
    count = 0
    for s in range(0,fshape[0]):
        for c in range(0,fshape[1]):

            shape = [len(results[count:count + points[s][c]])]

            trials = np.asarray(results[count:count + points[s][c]])

            # Get error from computation uncertainty
            if e_results != None:

                e_trials = np.asarray(e_results[count:count + points[s][c]])

            else:

                e_trials = np.zeros(trials.shape)

            [shape.append(i) for i in fshape[2:]]

            trials = trials.reshape(shape)
            e_trials = e_trials.reshape(shape)

            trials = np.concatenate((trials[:,np.newaxis], e_trials[:,np.newaxis]), axis = 1)

            count = count + points[s][c]

            RES.append(trials)

    return RES

# Prepare corrsum.py results for correxp.py script
def correxp_getcorrsum(path: str):

    # Load correlation sum results
    CS, _, variables = loadresults(obs_path = path, obs_name = 'corrsum')

    flat_CS, points = flat_results(CS)

    # Initzialize trial-wise iterable
    log_CS_iters = []
    for arr in flat_CS:

        log_CS = to_log(arr, verbose = False)

        # Build trial-wise iterable
        log_CS_iters.append([log_CS[0],log_CS[1]])

    return log_CS_iters, points, variables

# Prepare correxp.py results for peaks.py script
def peaks_getcorrexp(path: str):

    # Load correlation sum results
    CE, _, variables = loadresults(obs_path = path, obs_name = 'correxp')

    flat_CE, points = flat_results(CE)

    # Initzialize trial-wise iterable
    CE_iters = []
    for arr in flat_CE:

        # Build trial-wise iterable
        CE_iters.append([arr[0],arr[1]])

    return CE_iters, points, variables

### HELPER FUNCTIONS ###

# Euclidean distance
def dist(x, y, m_norm = False, m = None):

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
def td_embedding(ts, embedding: int, tau: int, fraction = None):

    # If time series isn't an array now it is
    ts = np.asarray(ts)
    
    # Get just a piece of it
    if fraction != None:
        ts = ts[int(fraction[0]*len(ts)):int(fraction[1]*len(ts))]

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

    return emb_ts

### OBSERVABLES FUNCTIONS ON EMBEDDED TIME SERIES ###

# Correlation sum for a single embeddend time series [Grassberger-Procaccia]
def corr_sum(emb_ts, r: float, m_norm = False, m = None):

    N = emb_ts.shape[-1]

    counter = 0

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i):

            dij = dist(emb_ts[:,i],emb_ts[:,j], m_norm = m_norm, m = m)

            # Get value of theta
            if dij < r:
                counter += 1

    csum = (2/(N*(N-1)))*counter

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

# Information Dimension for a single embedded time series with 2NN-estimation [Krakovská-Chvosteková]
def idim(emb_ts: np.ndarray, m_period: int):

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

#[IMPLEMENT BETTER]
def ce_plateau():

    a = trial_CE[0]
    a_ = trial_CE[1]

    # Initzialize results array
    D2 = []
    D2e = []
    D2r = []
    for a1, a1_ in zip(a,a_):
        for a2, a2_ in zip(a1,a1_):

            # Search for the plateau(Needs polishing)
            peaks = find_peaks(-a2, distance = 5)

            d2s = []
            r = []
            e = []
            for i in peaks[0]:
                #if log_r[i] < 0: #THIS IS ARBITRAY
                d2s.append(a2[i])
                e.append(a2_[i])
                r.append(log_r[i])

            D2.append(np.asarray(d2s).mean())
            D2e.append(np.asarray(d2s).std() + np.asarray(e).mean())
            D2r.append(np.asarray(r).mean())

    D2 = np.asarray(D2).reshape(a.shape[:-1])
    D2e = np.asarray(D2e).reshape(a.shape[:-1])
    #D2 = np.concatenate((D2[np.newaxis], D2e[np.newaxis]), axis = 0)

    D2r = np.asarray(D2r).reshape(a.shape[:-1])

    return D2, D2e, D2r

# Find peaks in Correlation Exponent trial-wise results
def ce_peaks(trial_CE: np.ndarray, log_r: list, distance: int, height: list, prominence: list, width: list):

    a = trial_CE[0]
    a_ = trial_CE[1]

    # Initzialize results array
    D2 = []
    D2e = []
    D2r = []
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

# Delay time of channel time series of a specific trial
def delay_time(evoked: mne.Evoked, ch_list: list|tuple,
               method: str, clst_method: str, fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    # Trim time series according to fraction variable
    tmin = times[int(fraction[0]*(len(times)-1))]
    tmax = times[int(fraction[1]*(len(times)-1))]

    evoked.crop(tmin = tmin, tmax = tmax)

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

# Correlation sum of channel time series of a specific trial
def correlation_sum(evoked: mne.Evoked, ch_list: list|tuple,
                    embeddings: list, tau: str|int, rvals: list, 
                    m_norm: bool, fraction = [0,1]):

    # Apply fraction to time series
    times = evoked.times

    # Trim time series according to fraction variable
    tmin = times[int(fraction[0]*(len(times)-1))]
    tmax = times[int(fraction[1]*(len(times)-1))]

    evoked.crop(tmin = tmin, tmax = tmax)

    # Initzialize result array
    CD = []

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

                emb_ts = np.asarray([emb_ts[i,j] for j in range(0,emb_ts.shape[1]) for i in range(0,emb_ts.shape[0])])

                for r in rvals:

                    CD.append(corr_sum(emb_ts, r = r, m_norm = m_norm, m = np.sqrt(len(emb_ts))))

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

                for r in rvals:

                    CD.append(corr_sum(emb_ts, r = r, tau = tau, m_norm = m_norm, m = np.sqrt(m)))

    # Returns list in -C style ordering

    return CD

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
