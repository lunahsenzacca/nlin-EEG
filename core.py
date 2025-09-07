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
def obs_data(obs_path: str, obs_name: str, compound_error = False):

    # Load result variables
    with open(obs_path + 'variables.json', 'r') as f:
        variables = json.load(f)

    if obs_name == 'corrsum':

        M = np.load(obs_path + 'corrsum.npy')
        X = np.load(obs_path + 'rvals.npy')

    if obs_name == 'idim':

        M = np.load(obs_path + 'idim.npy')
        X = variables['embeddings']

    elif obs_name == 'llyap':

        M = np.load(obs_path + 'llyap.npy')
        X = variables['embeddings']

    OBS = M[0]
    E_OBS = M[1]

    if compound_error == True:
        E_OBS = E_OBS + M[2]

    return OBS, E_OBS, X, variables

# Convert channel names to appropriate .mat data index
def name_toidx(names: list| tuple, exp_name: str):

    # Load the .mat file (adjust the path)
    mat_data = sio.loadmat(maind[exp_name]['directories']['ch_info'])

    # Get list of electrodes names in .mat file
    ch_list = []

    ### THIS PROBABLY IS DATASET SPECIFIC, BE CAREFUL WITH NEW DATA
    for m in mat_data['Channel'][0]:
        ch_list.append(str(m[0][0]))

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

    # Load the .mat file (Some folders could be missing it)
    mat_data = sio.loadmat(maind[exp_name]['directories']['ch_info'])

    # Get list of electrodes names in .mat file
    ch_list = maind[exp_name]['pois']

    # Get time points lenght
    Tst = maind[exp_name]['T']

    # Get conditions
    conditions = list(maind[exp_name]['conditions'].values())

    # Loop over conditions
    data_list = []
    for cond in conditions:
        
        # This depends on the raw data structure
        if exp_name == 'bmasking':
            my_cond_files = [f for f in all_files if cond in f ]
        
        # Get indexes of electrodes
        untup = name_toidx(ch_list, exp_name = exp_name)

        all_trials = np.empty((0,len(untup),Tst))
            
        for f in my_cond_files:

            path = sub_folder + f
            
            # This also depends on raw data structure
            if exp_name == 'bmasking':
                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][untup]

            all_trials = np.concatenate((all_trials, data[np.newaxis,:]), axis = 0)
        
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
    info.set_montage(maind[exp_name]['montage'])

    return info

# Function for subwise evoked conversion from list of arrays
def list_toevoked(data_list: list, subID: str, exp_name: str, method: str, alt_sv: str):
    
    # There are different number of trials for each condition,
    # so a simple ndarray is inconvenient. We use a list of ndarray instead

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

        # Average across same condition trials
        if method == 'avg_data':

            n_trials = array.shape[0]
            avg = array.mean(axis = 0)

            ev = mne.EvokedArray(avg, info, nave = n_trials, comment = conditions[i])
            evokeds.append(ev)

        # Keep each individual trial
        elif method == 'trl_data':

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
def toevoked(subID: str, exp_name: str, method: str, alt_sv = None):

    # Create data list
    data_list = raw_tolist(subID = subID, exp_name = exp_name)

    # Generate evokeds
    evokeds = list_toevoked(data_list = data_list, subID = subID, exp_name = exp_name, method = method, alt_sv = alt_sv)

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

# Function for trials averaging and homogeneous array generation (works for already averaged trials as well)
def collapse_trials(results: list, points: list, fshape: list, e_results = None):

    if e_results != None:
        print('\nObservable error in input, will be appended in results along trial error')

    # Initzialize homogeneous arrays for results and standard deviations
    RES = []
    RES_STD = []

    if e_results != None:
        OBS_STD = []

    # Make the array homogeneous
    count = 0
    for s in range(0,fshape[0]):
        for c in range(0,fshape[1]):

            trials = np.asarray(results[count:count + points[s][c]])

            # Average across trial results
            avg = np.mean(trials, axis = 0)
            std = np.std(trials, axis = 0)

            RES.append(avg)
            RES_STD.append(std)

            # Get error from computation uncertainty
            if e_results != None:

                e_trials = np.asarray(e_results[count:count + points[s][c]])
                e_avg = np.sqrt(np.sum(e_trials**2, axis = 0))/len(e_trials)

                OBS_STD.append(e_avg)

            count = count + points[s][c]
            
    RES = np.asarray(RES)
    RES = RES.reshape(fshape)
    
    RES_STD = np.asarray(RES_STD)
    RES_STD = RES_STD.reshape(fshape)

    if e_results != None:

        OBS_STD = np.asarray(OBS_STD)
        OBS_STD = OBS_STD.reshape(fshape)

        # Create a single array to store value and errors
        RES = np.concatenate((RES[np.newaxis],RES_STD[np.newaxis], OBS_STD[np.newaxis]), axis = 0)

    else:
        # Create a single array to store value and error
        RES = np.concatenate((RES[np.newaxis],RES_STD[np.newaxis]), axis = 0)

    return RES

### HELPER FUNCTIONS ###

# Euclidean distance
def dist(x, y):
    return np.sqrt(np.sum((x - y)**2, axis = 0))

# Transform data in log scale (Useful for logarithmic fits)
def to_log(CSums, rvals):

    # Get logarithmic scale
    log_CS = CSums.copy()
    log_r = np.log(rvals)

    # Embedding dimensions
    embs = [i for i in range(2,CSums.shape[3]+2)]
    c = 0
    # Substitute 0 values with nan values instead of whatever numpy is doing
    with np.nditer(log_CS, op_flags=['readwrite']) as it:
        for x in tqdm(it, desc = 'Getting logarithms',
                        total = it.shape[0], leave = False):
            
            if x == 0:
                c+=1
                x[...] = None
            else:
                x[...] = np.log(x)

    print('\nZero valued data points: ' + str(c))

    return log_CS, log_r

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

    return ts[idxs]

### OBSERVABLES FUNCTIONS ON EMBEDDED TIME SERIES ###

# Correlation sum for a single embeddend time series [Grassberger-Procaccia]
def corr_sum(emb_ts, r: float):

    N = len(emb_ts)

    ds = np.zeros((N,N), dtype = np.float64)

    counter = 0

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i):
            
            dij = dist(emb_ts[i],emb_ts[j])

            # Get value of theta
            if dij < r:
                counter += 1

    csum = (2/(N*(N-1)))*counter

    return csum

# Largest Lyapunov exponent for a single embedded time series [Rosenstein et al.]
def lyap(emb_ts, m_period: int, sfreq: int, verbose = False):

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

    # Duration of separation trajectories
    lenght = int(m_period/3)
    for i, el in enumerate(ds):

        if i < N - lenght:

            # Select only distant points on the trajectory but not too far on the end
            js = [j for j in range(0,N)]
            jt = []
            for j in js:
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

    fit = linregress(x,y)

    lyap = fit.slope
    lyap_e = fit.stderr

    return lyap, lyap_e, x, y, fit

### SUB-TRIAL WISE FUNCTIONS FOR OBSERVABLES COMPUTATION ###

# Correlation dimension of channel time series of a specific trial
def correlation_sum(evoked: mne.Evoked, ch_list: list|tuple,
                    embeddings: list, tau: int, rvals: list, 
                    fraction = [0,1]):

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

            for r in rvals:

                CD.append(corr_sum(emb_ts, r = r))

    # Returns list in -C style ordering

    return CD

# Largest lyapunov exponent of channel time series
def lyapunov(evoked: mne.Evoked, ch_list: list | tuple, 
             embeddings: list, tau: int,
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
        #f, ps = periodogram(ts)
        #avT = np.sum(ps)/np.sum(f*ps)

        # Or set a fixed lenght
        avT = tau*2
        
        for m in embeddings:
            emb_ts = td_embedding(ts, embedding = m, tau = tau)
            
            l, l_e, x, y, fit = lyap(emb_ts, m_period = avT, sfreq = sfreq, verbose = verbose)

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

### RESULTS MANIPULATION FUNCTION ###

# Select some electrodes from global results
def reduceCS(ch_list: list, path : str, label = 'G', nlabel = None):

    CS = np.load(path + label + '/corrsum.npy')
    r = np.load(path + label + '/rvals.npy')

    with open(path + label + '/variables.json', 'r') as f:
        d = json.load(f)

    # Check if we are loading a clusterd calculation [certaingly not suitable]
    if d['clustered'] == True:
        print('Clustered results are not valid for array reduction')
        return

    ch_idx = name_toidx(ch_list)

    shp = np.asarray(CS.shape)
    shp[2] = 0

    CSred = np.empty(shp)
    for idx in ch_idx:
        CSred = np.concatenate((CSred, CS[:,:,idx,:,:][:,:,np.newaxis,:,:]), axis = 2)

    # Save results in a new directory
    if nlabel != None:

        # Change dictionary entry for save    
        d['pois'] = ch_list

        os.makedirs(path + nlabel, exist_ok = True)
        np.save(path + nlabel + '/CSums.npy', CSred)
        np.save(path + nlabel + '/rvals.npy', r)

        # Save new variables dictionary
        with open(path + nlabel + '/variables.json', 'w') as f:
            json.dump(d, f)

    return CSred, r

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
