import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
import os

from time import time

from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.MI_delay import MI_for_delay

#Time series analysis parameters

#Time series lenght (--> implement autocheck)
Tst = 451

#Nearest neighbours for tau estimation through first MI minimum
k_set = 3

#Threshold for FNN detection
Rtol_set = 20

#Workflow folder path
path = '/home/lunis/Documents/nlin-EEG/'

#Backward Masking dataset path
bw_path = path + '/data/backward_masking/'
bw_pics_path = path + '/pics/backward_masking/'



#Convert a list into a tuple of lists
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

def twodim_graphs(subID: str, tau: int, trim: int, conditions: list, channels_idx: list | tuple):
    
    results = [subID]

    folder = bw_path + 'subj' + subID + '_band_resample/'
    all_files = os.listdir(folder)
    
    save_path = bw_pics_path + '/2dim_emb/'+ subID + '/'
    os.makedirs(save_path, exist_ok = True)

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
                #print(raw_ts.max(), raw_ts.min())
                idx_val = []
                for st in chs:
                    idx_val = idx_val + [int(st)]
                idx_str = str(idx_val)

                emb = np.empty((2, Tst - 2*trim - tau))

                #Embedd time series
                emb[0,:] = raw_ts[trim:Tst - trim - tau]
                emb[1,:] = raw_ts[trim + tau : Tst - trim]
                
                fig, ax = plt.subplots()
                ax.plot(emb[0,:], emb[1,:], 'ko-', linewidth = 1, markersize = 3)

                ax.set(title = cond + idx_str)

                plt.savefig(save_path + cond + idx_str + '.png')
                plt.close()

        elif type(channels_idx) == list:

            all_trials = np.empty((0,Tst))

            for f in my_cond_files:

                path = folder+f

                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][channels_idx]
                all_trials = np.concatenate((all_trials, data.mean(axis=0)[np.newaxis, :]))

            #Get the time series data (Average across trials)
            raw_ts = all_trials.mean(axis=0)

            idx_val = []
            for chs in channels_idx:

                idx_val = idx_val + [int(chs)]
            
            idx_str = str(idx_val)

            emb = np.empty((2, Tst - 2*trim))

            for t in range(trim, Tst - tau - trim):

                emb[0,t - trim] = raw_ts[t]
                emb[1,t - trim] = raw_ts[t + tau]

            fig, ax = plt.subplots()
            ax.plot(emb[0,:], emb[1,:], 'ko-', linewidth = 1, markersize = 3)

            ax.set(title = cond + idx_str)

            plt.savefig(save_path + cond + idx_str + '.png')
            plt.close()

        else:

            print('Channel indexes format error, check channel_idx data type:\n' + str(type(channels_idx)))
            return   

    #print('Sub' + subID + ': Done! ')

    return


# Return array with min and max value taken by the POIs average time series

#   This is done to check for datas anomalies

def check_maxmin(subID: str, conditions: list, channels_idx: list | tuple):
    
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

                smax = np.max(raw_ts)
                smin = np.min(raw_ts)
                results = results + [smax, smin]

        elif type(channels_idx) == list:

            all_trials = np.empty((0,Tst))

            for f in my_cond_files:

                path = folder+f

                mat_data = sio.loadmat(path)
                    
                data = mat_data['F'][channels_idx]
                all_trials = np.concatenate((all_trials, data.mean(axis=0)[np.newaxis, :]))

            #Get the time series data (Average across trials)
            raw_ts = all_trials.mean(axis=0)
            
            smax = np.max(raw_ts)
            smin = np.min(raw_ts)
            results = results + [smax, smin]

        else:

            print('Channel indexes format error, check channel_idx data type:\n' + str(type(channels_idx)))
            return   
            
    print('Sub' + subID + ': Done! ')

    #Remebrer the hierarchy! If it applies
    return results


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
