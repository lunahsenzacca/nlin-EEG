import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
import os

from time import time

from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.MI_delay import MI_for_delay

Tst = 451
Rtol_set = 10
k_set = 3
bw_path = '/home/lunis/Documents/EEG/data/backward_masking/'
c_path = '/home/lunis/Documents/EEG/data/calc/'

#Convert a list into a tuple of lists

def tuplinator(list):

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

#M

def init_Frame(conditions, ch_clust_idx, metrics):

    df_cols = ['SubID']

    for cnd in conditions:

        for chs in ch_clust_idx:

            if type(chs) == list:

                idx_str = []
                for st in chs:
                    idx_str = idx_str + [int(st)]
                idx_str = str(idx_str)

            elif type(chs) == np.int64:

                idx_str = [int(chs)]
                idx_str = str(idx_str)

            else:
                    
                print('Channel indexes format error, check channel_idx data type:\n' + str(type(chs)))
                print('Functions that take it could break!')
        
            for met in metrics:

                df_cols = df_cols + [met + ' ' + cnd + ' ' + idx_str]

    df = pd.DataFrame(columns = df_cols)

    return df

#Main processing

def SSub_process(subID, conds, channels_idx, DataFrame):
    
    results = [subID]

    folder = bw_path + 'subj' + subID + '_band_resample/'
    all_files = os.listdir(folder)
    # cond = 'S_1' # unconscious

    # Loop over conditions (conscious and unconscious)
    for cond in conds:
        
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
                
    print(subID + ': Done!')
    return results
