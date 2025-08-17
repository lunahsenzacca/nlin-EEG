import os
import mne

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.MI_delay import MI_for_delay
from teaspoon.SP.information.entropy import PE

from scipy.stats import linregress

from multiprocessing import Pool

from tqdm import tqdm

# Time series analysis parameters

# Time series lenght (--> implement autocheck)
Tst = 451

# Frequency
freq = 500

# Nearest neighbours for tau estimation through first MI minimum
k_set = 3

# Threshold for FNN detection
Rtol_set = 15

# Workflow folder path
path = '/home/lunis/Documents/nlin-EEG/'

# Backward Masking dataset path
bw_path = path + '/data/backward_masking/'
bw_pics_path = path + '/pics/backward_masking/'

#####################################

# Utility functions

# Get subject path
def sub_path(subID: str, experiment = 'bw'):

    if experiment == 'bw':

        path = bw_path + 'subj' + subID + '_band_resample/'

    return path

# Convert channel names to appropriate .mat data index
def name_toidx(names: list| tuple):
    # Load the .mat file (adjust the path)
    mat_data = sio.loadmat(bw_path + 'subj001_band_resample/channel.mat')

    ch_list = []
    for m in mat_data['Channel'][0]:
    #     print(m[0][0])
        ch_list.append(str(m[0][0]))

    if type(names) ==  tuple:

        first = True
        for c in names:
            part = []
            
            for sc in c:
            
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


# Convert subject data to evoked format for easier manipulation
def mat_toevoked(subID: str, conditions: list, exp: str, freq: float):

    sub_folder = sub_path(subID, experiment = exp)
    all_files = os.listdir(sub_folder)

    # Load the .mat file (Some folders could be missing it)
    mat_data = sio.loadmat(sub_path('001', experiment =  exp) +'channel.mat')

    ch_list = []
    for m in mat_data['Channel'][0]:
    #     print(m[0][0])
        ch_list.append(str(m[0][0]))

    # Create info file
    ch_types = ['eeg' for n in range(0, len(ch_list))]

    inf = mne.create_info(ch_list, ch_types = ch_types, sfreq = freq)
    inf.set_montage('standard_1020')

    # Initzialize evoked list
    evoked = []

    # Loop over conditions
    for cond in conditions:
        
        my_cond_files = [f for f in all_files if cond in f ]
            
        untup = name_toidx(ch_list)

        all_trials = np.empty((0,len(untup),Tst))
            
        for f in my_cond_files:

            path = sub_folder+f
                
            mat_data = sio.loadmat(path)
                
            data = mat_data['F'][untup]

            
            all_trials = np.concatenate((all_trials, data[np.newaxis,:]), axis = 0)
        
        n_trials = all_trials.shape[0]
        signals = all_trials.mean(axis = 0)

        evoked.append(mne.EvokedArray(signals, inf, nave = n_trials, comment = cond))

    #print('Sub' + subID + ': Done! ')
    return evoked

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

#####################################

# Helper functions

# Euclidean distance
def dist(x, y):
    return np.sqrt(np.sum((x - y)**2, axis = 0))

# Transform data in log scale
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
                        total = it.shape[0], leave = True):
            
            if x == 0:
                c+=1
                x[...] = None
            else:
                x[...] = np.log(x)

    print('Zero valued data points: ' + str(c))

    return log_CS, log_r

#####################################

# Time series manipulation function

# Time-delay embedding
def td_embedding(ts, embedding: int, tau: int, fraction = None):

    ts = np.asarray(ts)
    
    # Get just a piece of it
    if fraction != None:
        ts = ts[int(fraction[0]*len(ts)):int(fraction[1]*len(ts))]

    min_len = (embedding - 1)*tau + 1

    if len(ts) < min_len:

        print('Data lenght is insufficient, try smaller parameters')
        return
    
    # Set lenght of embedding
    m = len(ts) - min_len + 1

    # Get indeces
    idxs = np.repeat([np.arange(embedding)*tau], m, axis = 0)
    idxs += np.arange(m).reshape((m, 1))

    return ts[idxs]

#####################################

# Observables functions

# Correlation sum
def corr_sum(emb_ts, r: float):

    N = len(emb_ts)

    ds = np.zeros((N,N), dtype = np.float64)

    counter = 0

    for i in range(0,N):
        for j in range(0,i):
            
            dij = dist(emb_ts[i],emb_ts[j])
            ds[i,j] = dij
            ds[j,i] = dij

            if dij < r:
                counter += 1

    csum = (2/(N*(N-1)))*counter

    return csum

# Compute average permutation entropy of a specific time series
def perm_entropy(subID: str, ch_list: list, conditions: list, embedding: int, ac_time: int, pth: str):

    # Implement directly from evoked data instead of averaging again across trials

    return PE

# Correlation dimension of channel time series 
def correlation_sum(subID: str, ch_list: list, conditions: list,
                    embeddings: list, tau: int, rvals: list, fraction: list,
                    pth: str):

    # Get evokeds
    file = pth + subID + '-ave.fif'
    evokeds = mne.read_evokeds(file, verbose = False)
    
    # Get only conditions of interest
    for e in evokeds:
        if e.comment not in conditions:
            evokeds.remove(e)

        times = e.times

        tmin = times[int(fraction[0]*(len(times)-1))]
        tmax = times[int(fraction[1]*(len(times)-1))]

        e.crop(tmin = tmin, tmax = tmax)

    # Start looping around
    CD = np.empty((0,len(ch_list),len(embeddings),len(rvals)))
    for i, cond in enumerate(conditions):

        TS = evokeds[i].get_data(picks = ch_list)

        CD1 = np.empty((0,len(embeddings),len(rvals)))
        for ts in TS:

            CD2 = np.empty((0,len(rvals)))
            for m in embeddings:
                emb_ts = td_embedding(ts, embedding = m, tau = tau)

                cd = []
                for r in rvals:

                    cd.append(corr_sum(emb_ts, r = r))

                CD2 = np.concatenate((CD2, np.asarray(cd)[np.newaxis,:]), axis = 0)
    
            CD1 = np.concatenate((CD1, CD2[np.newaxis,:,:]), axis = 0)

        CD = np.concatenate((CD, CD1[np.newaxis,:,:,:]), axis = 0)

    return CD





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
