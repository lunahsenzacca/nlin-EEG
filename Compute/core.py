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
import io
import zipfile
import mne
import json
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

from time import time

from rich.progress import Progress, track

from multiprocessing import Pool

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

# Compute cutoff for Persistence Features (Not Used For Now...)
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

# Squareform and distance matrix functions
from scipy.spatial.distance import squareform, pdist

### UTILITY FUNCTIONS ###

# Cython modules compilation
def cython_compile(verbose: bool, setup_name: str = 'cython_setup'):

    print('\nCompiling cython file...')

    with warnings.catch_warnings():

        if verbose is True:

            warnings.simplefilter('ignore')

        os.system(f'python ./cython_modules/{setup_name}.py build_ext -b ./cython_modules/ -t ./cython_modules/build/')

    return

# Get subject path
def sub_path(subID: str, exp_name: str) -> str:

    # Text to attach before and after subID to get subject folder name for raw data
    snip = maind[exp_name]['directories']['subject']
    
    path = snip[0] + subID + snip[1]

    return path

# Get observable path
def obs_path(exp_name: str, obs_name: str, clst_lb: str, avg_trials: bool, calc_lb: str | None = None) -> str:

    if avg_trials == True:
        results = 'avg_results'
    else:
        results = 'trl_results'

    # Create directory string
    path = maind[exp_name]['directories'][results] + clst_lb + '/'+ maind['obs_lb'][obs_name] + '/'

    if calc_lb != None:

        path += calc_lb + '/'

    return path

# Get pics path
def pics_path(exp_name: str, obs_name: str, clst_lb: str, avg_trials: bool, calc_lb: str | None = None) -> str:

    if avg_trials == True:
        pics = 'avg_pics'
    else:
        pics = 'trl_pics'

    # Create directory string
    path = maind[exp_name]['directories'][pics] + clst_lb + '/'+ maind['obs_lb'][obs_name] + '/'

    if calc_lb != None:

        path += calc_lb + '/'

    return path

# Get observable data
def obs_data(obs_path: str, obs_name: str) -> tuple[np.lib.npyio.NpzFile, list, dict]:

    # Load result info
    with open(obs_path + 'info.json', 'r') as f:
        info = json.load(f)

    if obs_name == 'evokeds':

        M = np.load(obs_path + 'evokeds.npz')
        x = info['t']

        X = [x]

    elif obs_name == 'spectrum':

        M = np.load(obs_path + 'spectrum.npz')
        x = info['freqs']

        X = [x]

    elif obs_name == 'delay':

        M = np.load(obs_path + 'delay.npz')
        x = info['ch_list']

        X = [x]

    elif obs_name == 'recurrence':

        M = np.load(obs_path + 'recurrence.npz')
        x = info['t']

        X = [x]

    elif obs_name == 'separation':

        M = np.load(obs_path + 'separation.npz')
        x = info['embeddings']
        y = info['dt']

        X = [x,y]

    elif obs_name == 'corrsum':

        M = np.load(obs_path + 'corrsum.npz')
        x = info['embeddings']
        y = info['log_r']

        X = [x,y]

    elif obs_name == 'correxp':

        M = np.load(obs_path + 'correxp.npz')
        x = info['embeddings']
        y = info['log_r']

        X = [x,y]

    elif obs_name == 'peaks':

        M = np.load(obs_path + 'peaks.npz')
        x = info['embeddings']

        # See if you want to implement it later
        #y = np.load(obs_path + 'peaks_r.npz')

        X = [x]

    elif obs_name == 'plateaus':

        M = np.load(obs_path + 'plateaus.npz')
        x = info['embeddings']
        y = info['log_r']

        # See if you want to implement it later
        z = np.load(obs_path + 'plateaus_r.npz')

        X = [x,y,z]

    elif obs_name == 'idim':

        M = np.load(obs_path + 'idim.npz')
        x = info['embeddings']

        X = [x]

    elif obs_name == 'llyap':

        M = np.load(obs_path + 'llyap.npz')
        x = info['embeddings']

        X = [x]

    return M, X, info

# Convert channel names to appropriate .mat data index
def name_toidx(names: list | tuple, exp_name: str) -> list:
    
    # Get list of electrodes names
    ch_list = maind[exp_name]['pois']

    # Check if we are clstering electrodes with tuples
    if type(names) ==  tuple or type[names[0]] == list:

        first = True
        for c in names:
            part = []
            
            for sc in c:
                
                # Add new indexes to clster list
                part = part + [np.where(np.asarray(ch_list)==sc)[0][0]]

            part = part,
            if first == True:

                ch_clst_idx = part
                first = False

            else:
                ch_clst_idx = ch_clst_idx + part
                

    else:

        ch_clst_idx = []
        for c in names:
            ch_clst_idx.append(np.where(np.asarray(ch_list)==c)[0][0])

    return ch_clst_idx

# Convert a list into a tuple of lists
def tuplinator(list: list) -> tuple:

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
def get_tinfo(exp_name: str, avg_trials: bool, window: list = [None,None]) -> tuple[dict, list]:

    if avg_trials == True:
        method = 'avg_data'
    else:
        method = 'trl_data'

    # Evoked folder paths
    path = maind[exp_name]['directories'][method]

    # List of ALL subject IDs
    sub_list = maind[exp_name]['subIDs']

    # List of of conditions
    conditions = list(maind[exp_name]['conditions'].values())
    
    # Point to first subject and first condition
    fpath = path + sub_list[0] + '_' + conditions[0]

    if method == 'avg_data':
        data = mne.read_evokeds(fpath + '-ave.fif', verbose = False)[0]
    else:
        data = mne.read_epochs(fpath + '-epo.fif', verbose = False)

    data.crop(tmin = window[0], tmax = window[1], include_tmax = False)

    times = data.times

    info = data.info

    return info, times

# Function for single subject conversion from raw data to list  for MNE 
def raw_tolist(subID: str, exp_name: str) -> tuple[list, list, list]:

    # Navigate subject folder with raw single trial files or epochs files
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
    info = []
    events = []
    for cond in conditions:

        my_cond_files = [f for f in all_files if cond in f ]

        # Get indexes of electrodes
        untup = name_toidx(ch_list, exp_name = exp_name)

        all_trials = np.empty((0,len(untup),Tst)) 

        c_events = np.empty((0,3))
        for f in my_cond_files:

            path = sub_folder + f
            
            # This also depends on raw data structure
            if exp_name == 'bmasking' or exp_name == 'zbmasking':
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup][np.newaxis]

                # There is probably a way to fix this
                info.append(None)
                c_events = np.concatenate((c_events, np.array([None, None, None])[np.newaxis]), axis = 0)

            else:

                mne_data = mne.read_epochs(path, preload = True, verbose = False)

                data = mne_data.get_data(picks = untup)

                info.append(mne_data.info)

                c_events = np.concatenate((c_events, mne_data.events), axis = 0)

            all_trials = np.concatenate((all_trials, data), axis = 0)

        data_list.append(all_trials)

        sorts = np.argsort(c_events[:,0])

        c_events = np.asarray([c_events[i] for i in sorts], dtype = np.int32)

        events.append(c_events)

    return data_list, info, events

# Create info file for specific datased
def toinfo(exp_name: str, info: mne.Info, ch_type: str = 'eeg') -> mne.Info:

    if type(info) == None:

        # Get electrodes labels
        ch_list = maind[exp_name]['pois']

        ch_types = [ch_type for ch in ch_list]

        # Get sampling frequency
        freq = maind[exp_name]['f']

        info = mne.create_info(ch_list, ch_types = ch_types, sfreq = freq)

    return info

# Function for subwise MNE conversion from list of arrays
def list_toMNE(data_list: list, info: mne.Info, events: tuple, subID: str, exp_name: str, avg_trials: bool, z_score: bool, baseline: bool, sv_path: str):
    
    # There are different number of trials for each condition,
    # so we cannot make one huge homogeneous ndarray, we have to save
    # different files per subject and per condition

    # The list index cycles faster around the conditions and slower around the subject

    # Create info objects
    info = toinfo(exp_name = exp_name, info = info)

    # Get loaded starting time
    tmin = maind[exp_name]['window'][0]

    conditions = list(maind[exp_name]['conditions'].values())

    # Cycle around conditions
    for i, array in enumerate(data_list):

        # File name for saving
        fname = sv_path + subID + '_' + conditions[i]

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

            ev = mne.EvokedArray(avg, info[i], nave = n_trials, tmin = tmin, kind = 'average', verbose = False)
            s_ev = mne.EvokedArray(std, info[i], nave = n_trials, tmin = tmin, kind = 'standard_error', verbose = False)
            
            ev.save(fname + '-ave.fif', overwrite = True, verbose = False)
            s_ev.save(fname + '-std-ave.fif', overwrite = True, verbose = False)

            del ev, s_ev, array_

        # Or keep each individual trial
        else:

            ep = mne.EpochsArray(array_, info[i], events = events[i], tmin = tmin, verbose = False)

            ep.save(fname + '-epo.fif', overwrite = True, verbose = False)

            del ep, array_

    return

# Create evoked file straight from raw data
def toMNE(subID: str, exp_name: str, avg_trials: bool, sv_path: str, z_score: bool = False, baseline: bool = False):

    # Create data list
    data_list, info, events = raw_tolist(subID = subID, exp_name = exp_name)

    # Generate evokeds or epochs and save them
    list_toMNE(data_list = data_list, info = info, events = events, subID = subID, exp_name = exp_name, avg_trials = avg_trials, z_score = z_score, baseline = baseline, sv_path = sv_path)

    return

# Load MNE files of a specific subject
def loadMNE(exp_name: str, avg_trials: bool, subID: str, conditions: list, with_std: bool = False) -> list:

    # Select correct path for data
    if avg_trials == True:

        file_p = maind[exp_name]['directories']['avg_data']

        ext = '-ave.fif'
        if with_std == True:
            s_ext = '-std' + ext

    else:
        if with_std == True:
            raise ValueError('Asking for an Epoch Standard Error object, this makes no sense!')

        file_p = maind[exp_name]['directories']['trl_data']

        ext = '-epo.fif'

    # Create  
    full_data = []
    for cond in conditions:
        
        path = file_p + subID + '_' + cond + ext
        
        if avg_trials == True:

            data = mne.read_evokeds(path, verbose = False)[0]

            if with_std == True:

                s_path = file_p + subID + '_' + cond + s_ext

                s_data = mne.read_evokeds(s_path, verbose = False)[0]

                data = [data,s_data]

            else:

                data = [data]
        
        else:

            data = [mne.read_epochs(path, verbose = False)]

        full_data.append(data)

    return full_data

# Extract time series from MNE data structure in a convenient manner
# Returns a nested list of time series with the following hierarchy
# 
#   [Trials][Clusters][Electrodes in cluster][Time points]
# 
# Each function needs 2 for loops to get to the individual time series 1-d vector, just 1 for the cluster
def extractTS(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list | tuple, sMNE: mne.Evoked | None = None, window: list = [None, None], clst_method: str = 'append') -> tuple[list, list]:

    # Apply fraction to evoked objects
    MNE.crop(tmin = window[0], tmax = window[1], include_tmax = False)

    if sMNE != None:
        sMNE.crop(tmin = window[0], tmax = window[1], include_tmax = False)

    fTS = []
    E_fTS = []

    # Check if we are clstering electrodes
    if type(ch_list) == tuple or type(ch_list[0]) == list:

        for cl in ch_list:

            # Get average time series of the clster
            ts = MNE.get_data(picks = cl) 

            if sMNE != None:
                e_ts = sMNE.get_data(picks = cl)
            else:
                e_ts = np.zeros(ts.shape)

            if len(ts.shape) < 3:
                ts = ts[np.newaxis]
                e_ts = e_ts[np.newaxis]

            if clst_method == 'mean':
                ts = ts.mean(axis = 1)[:,np.newaxis]
                e_ts = e_ts.mean(axis = 1)[:,np.newaxis]

            fTS.append(ts)
            E_fTS.append(e_ts)

    else:

        for poi in ch_list:

            ts = MNE.get_data(picks = poi)

            if sMNE != None:
                e_ts = sMNE.get_data(picks = poi)
            else:
                e_ts = np.zeros(ts.shape)

            if len(ts.shape) < 3:
                ts = ts[np.newaxis]
                e_ts = e_ts[np.newaxis]

            fTS.append(ts)
            E_fTS.append(e_ts)

    n_trials = len(fTS[0])

    TS = [[] for i in range(0,n_trials)]
    E_TS = [[] for i in range(0,n_trials)]

    for i in range(0,n_trials):
        for c in range(0,len(ch_list)):
            TS[i].append(fTS[c][i])
            E_TS[i].append(E_fTS[c][i])

    return TS, E_TS

# Open results .npz file and convert it to nested list structure
def load_results(obs_path: str, obs_name: str, X_transform: int | None = None) -> tuple[list, list, dict]:

    M, X, info = obs_data(obs_path = obs_path, obs_name = obs_name)

    len_s = len(info['sub_list'])
    len_c = len(info['conditions'])

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

    return results, X, info

# Load last computed results
def load_last(X_transform: int | None = None) -> tuple[list, list, dict]:

    with open('.tmp/last.json','r') as f:

        info = json.load(f)

    path = obs_path(exp_name = info['exp_name'],
                    obs_name = info['obs_name'],
                    clst_lb = info['clst_lb'],
                    avg_trials = info['avg_trials'],
                    calc_lb = info['calc_lb'])

    results, X, info = load_results(obs_path = path, obs_name = info['obs_name'], X_transform = X_transform)

    return results, X, info

# Save numpy array in temporary path and return the latter as a string
def to_disk(arr: np.ndarray, sv_file: str):

    bio = io.BytesIO()

    np.save(bio, arr)

    with np.load(sv_file) as M:

        if len(M.files) == 0:

            id = 0

        else:

            id = int(M.files[-1].split('_')[1]) + 1

    with zipfile.ZipFile(sv_file, 'a',
                         compression = zipfile.ZIP_DEFLATED,
                         compresslevel = 1) as zipf:

        zipf.writestr(f'arr_{id}.npy', data=bio.getbuffer().tobytes(),
                      compress_type = zipfile.ZIP_DEFLATED,
                      compresslevel = 1)

    return

# Load list of numpy objects from list of paths
def from_disk(sv_file: str) -> list:

    data = []

    with np.load(sv_file) as M:

        for file in M.files:
            data.append(M[file])

    return data

# Create one dimensional list of results per subject per condition
def flat_results(results: list) -> list:

    # Flatten results nested list
    flat = [x for xss in results for xs in xss for x in xs]

    return flat

#  Function for homogeneous array saving after multiprocessing computation from disk or ram partially saved results
def save_results(results: list, fshape: list, info: dict, sv_name: str, e_results: list | None = None, dtype: type = np.float64, memory_safe: bool = False):

    # Generate result saving path
    sv_path = obs_path(exp_name = info['exp_name'], obs_name = info['obs_name'], avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = info['calc_lb'])

    sv_file = sv_path + sv_name + '.npz'

    os.makedirs(sv_path, exist_ok = True)

    # Initzialize list of results
    RES = []

    if memory_safe is True:

        # Initzialize compressed archive
        np.savez_compressed(sv_file)

    print('')

    # Make homogeneous arrays for each subject-condition combination
    for count in track(range(0,fshape[0]*fshape[1]),
                       description = '[red]Compressing results:',
                       total = fshape[0]*fshape[1],
                       transient = True):

        if memory_safe is True:

            sub_cond = from_disk(results[count])

        else:

            sub_cond = results[count]

        n_trials = len(sub_cond)//np.prod(fshape[2:-1])

        trials = np.asarray(sub_cond, dtype = dtype)

        shape_ = [n_trials, *fshape[2:]]

        trials = trials.reshape(shape_)

        # Get error from computation uncertainty
        if e_results != None:

            if memory_safe is True:

                e_sub_cond = from_disk(e_results[count])

            else:

                e_sub_cond = e_results[count]

            e_trials = np.asarray(e_sub_cond, dtype = dtype)

            e_trials = e_trials.reshape(shape_)

            trials = np.concatenate((trials[:,np.newaxis], e_trials[:,np.newaxis]), axis = 1)

        else:

            trials = trials[:,np.newaxis]

        if memory_safe is True:

            bio = io.BytesIO()

            np.save(bio, trials)

            with zipfile.ZipFile(sv_file, 'a', compression = zipfile.ZIP_DEFLATED) as zipf:

                zipf.writestr(f'arr_{count}.npy', data=bio.getbuffer().tobytes(), compress_type = zipfile.ZIP_DEFLATED)

        else:

            RES.append(trials)

    if memory_safe is False:

        np.savez_compressed(sv_file, *RES)

    # Save info file
    with open(sv_path + 'info.json','w') as f:
        json.dump(info, f, indent = 2)

    print('Results common shape: ', fshape[2:])

    return

# Prepare correxp.py results for peaks.py script
def pp_getcorrexp(path: str):

    # Load correlation sum results
    CE, _, variables = load_results(obs_path = path, obs_name = 'correxp', X_transform = None)

    flat_CE = flat_results(CE)

    # Initzialize trial-wise iterable
    CE_iters = []
    for arr in flat_CE:

        # Build trial-wise iterable
        CE_iters.append([arr[0],arr[1]])

    return CE_iters, variables

# Prepare recurrence.py results for corrsum.py script
def corrsum_getrecurrence(path: str):

    # Load correlation sum results
    RP, _, variables = load_results(obs_path = path, obs_name = 'recurrence', X_transform = None)

    flat_RP = flat_results(RP)

    # Initzialize trial-wise iterable
    RP_iters = []
    for arr in flat_RP:

        # Build trial-wise iterable
        RP_iters.append([arr[0],arr[1]])

    return RP_iters, variables

### HELPER FUNCTIONS ###

# Euclidean distance
def dist(x: np.ndarray, y: np.ndarray, m_norm: bool = False, m: int | None = None) -> float:

    d = np.sqrt(np.sum((x - y)**2, axis = 0))

    if m_norm and m != None:

        d /= m

    return d

# Transform data in log scale (Useful for logarithmic fits)
def to_log(OBS: np.ndarray, verbose: bool) -> np.ndarray:

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
def avg_period(ts: list) -> float:

    f, ps = periodogram(ts)
    avT = np.sum(ps)/np.sum(f*ps)

    return avT

# Get Lorenz attractor derivatives
def lorenz_delta(xyz, s: float = 10, r: float = 28, b: float = 8/3) -> np.ndarray:
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
def lorenz_trajectory(dt: float, time_points: int, target_l: int | None = None, x0: np.ndarray | None = None) -> np.ndarray:

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
def downsample(ts: list | np.ndarray, target_l: int) -> np.ndarray:

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
def zscore(trials_array: np.ndarray, keep_relations: bool = False) -> np.ndarray:
    
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
        z_trials_array = (trials_array - m)/s

    # Full electrode-wise Z-Score
    else:
        # Faster and more memory efficient
        M = np.broadcast_to(m[np.newaxis,:,np.newaxis], [trials_array.shape[0],len(m),trials_array.shape[2]])
        S = np.broadcast_to(s[np.newaxis,:,np.newaxis], [trials_array.shape[0],len(s),trials_array.shape[2]])

        # Apply Z-Score
        z_trials_array = np.divide(trials_array - M, S)

    return z_trials_array

# 1-D function Adymensional Gaussian kernel convolution
def gauss_kernel(function: np.ndarray, x: np.ndarray, scale: float, cutoff: int, order: int) -> np.ndarray:

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
            raise ValueError('Order 1 kernel Not yet implemented')

        c_function[i] = np.sum(vals*ker)/np.sum(ker)

    return c_function

### TIME SERIES MANIPULATION FUNCTIONS ###

# Time-delay embedding of a single time series (1-d vector)
def td_embedding(ts: np.ndarray, embedding: int, tau: str | int) -> np.ndarray:

    # Compute tau with given string method

    if tau == 'mutual_information':
        tau = int(MI_for_delay(ts))

    elif tau == 'autocorrelation':
        tau = int(autoCorrelation_tau(ts))

    else:
        if type(tau) != int:
            raise ValueError('tau method not \'mutual_information\' or \'autocorrelation\' or int')

    min_len = (embedding - 1)*tau + 1

    # Check if embedding is possible
    if len(ts) < min_len:

        raise ValueError('Data lenght is insufficient, try smaller parameters')

    # Set lenght of embedding
    m = len(ts) - min_len + 1

    # Get indexes
    idxs = np.repeat([np.arange(embedding)*tau], m, axis = 0)
    idxs += np.arange(m).reshape((m, 1))

    emb_ts = ts[idxs]

    emb_ts = np.asarray(emb_ts, dtype = np.float64)

    emb_ts = np.swapaxes(emb_ts, 0, 1)

    return emb_ts

# Time-delay embedding of a list of time series (n-d n>1 vector)
def multi_embedding(c_ts: list, embedding: int, tau: str | int) -> np.ndarray:

    lenghts = []
    emb_ts = []
    for ts in c_ts:

        emb_t = td_embedding(ts = ts, embedding = embedding, tau = tau)

        lenghts.append(emb_t.shape[1])
        emb_ts.append(emb_t)

    min_len = np.min(np.asarray(lenghts))

    emb_ts = np.asarray([t[0:min_len] for emb_t in emb_ts for t in emb_t ], dtype = np.float64)

    return emb_ts

### OBSERVABLES FUNCTIONS ON EMBEDDED TIME SERIES ###

def distance_matrix(emb_ts: np.ndarray, m_norm: bool = False, m: int | None = None) -> np.ndarray:

    emb_ts = emb_ts.T

    dist_matrix = squareform(pdist(emb_ts, metric="euclidean"))

    if m_norm and (m is not None):
        dist_matrix /= m

    return dist_matrix

