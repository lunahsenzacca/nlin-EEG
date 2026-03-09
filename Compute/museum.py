### OLD DEPRECATED OR STRAIGHT UP BAD STUFF THAT I DON'T WANT TO LOAD BUT STILL DON'T WANT TO DELETE ###

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

                cs = rec_corr_sum(recurrence_plot = a, w = w)

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


# NEEDS REVISING # Information dimension of channel time series of a specific trial
def information_dimension(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list|tuple,
                          embeddings: list, tau: int, 
                          window = None):

    # Apply fraction to time series
    if type(window) == list:   
        MNE.crop(tmin = window[0], tmax = window[1], include_tmax = False)

    # Initzialize result array
    D2 = []
    e_D2 = []

    # Check if we are clstering electrodes
    if type(ch_list) == tuple:
        

        tl = []
        for cl in ch_list:
            
            # Get average time series of the clster
            ts = MNE.get_data(picks = cl)
            ts = ts.mean(axis = 0)
            
            tl.append(ts)
        
        TS = np.asarray(tl)

    else:    

        TS = MNE.get_data(picks = ch_list)
    
    # Loop around pois time series
    for ts in TS:
        
        for m in embeddings:
            emb_ts = td_embedding(ts, embedding = m, tau = tau)

            d2, e_d2 = idim(emb_ts, m_period = tau)

            D2.append(d2)
            e_D2.append(e_d2)

    return D2, e_D2

def recursive_len(item):
    if type(item) == list or type(item) == np.ndarray:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1

### RESULTS MANIPULATION FUNCTION ###

# Select some electrodes from global results
def reduceCS(exp_name: str, avg_trials: bool, ch_list: list, average = False, clst_lb = 'G', nlabel = None):

    path = obs_path(exp_name = exp_name, clst_lb = clst_lb, obs_name = 'corrsum', avg_trials = avg_trials)

    CS = np.load(path + 'corrsum.npy')

    with open(path + 'variables.json', 'r') as f:
        d = json.load(f)

    log_r = d['log_r']

    # Check if we are loading a clsterd calculation [certaingly not suitable]
    if d['clstered'] == True:
        print('clstered results are not valid for array reduction')
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

        sv_path = obs_path(exp_name = exp_name, clst_lb = nlabel, obs_name = 'corrsum', avg_trials = avg_trials)

        os.makedirs(sv_path, exist_ok = True)
        np.save(sv_path + 'corrsum.npy', CSred)

        # Save new variables dictionary
        with open(sv_path + 'variables.json', 'w') as f:
            json.dump(d, f)

    return CSred, log_r

#   Following functions do not implement mne data structure and each of them averages over trials
#   They are useful because they can average over clster of electrodes as well but this doesn't seem
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
