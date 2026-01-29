import mne

import warnings

import numpy as np

from scipy.stats import linregress
from scipy.signal import periodogram

from core import extractTS, multi_embedding, distance_matrix

# Largest Lyapunov exponent for a distance matrix [Rosenstein et al.]
def lyap(dist_matrix: np.ndarray, dt: int, w: int, sfreq: int, T: int, verbose = False):

    ds = dist_matrix

    # Construct separations trajectories with embedded data
    lnd = []
    for i, el in enumerate(ds):

        if i < T - w:

            # Select only distant points on the trajectory but not too far on the end
            jt = []
            for j in range(0,T):
                if abs(j-i) > dt and j < T - w:
                    jt.append(j)

            if len(jt) != 0:
                # Get nearest neighbour
                d0 = np.min(el[jt])
                js = int(np.argwhere(el == d0))

            # Construct separation data
            for delta in range(0,w):
                if len(jt) != 0:
                    lnd.append(np.log(ds[i +delta,js + delta]))
                else:
                    lnd.append(np.nan)

    # Reshape results
    lnd = np.asarray(lnd).reshape((T - w,w))

    # Get slope for largest Lyapunov exponent
    x = np.asarray([i for i in range (0,w)])

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

# Largest Lyapunov Exponent of channel time series of a specific trial
def lyapunov(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list | tuple, 
             embeddings: list, tau: str|int, w: int, dt = None,
             m_norm = False, window = [None,None], cython = False, verbose = False):

    if cython == True:

        from cython_modules import c_core

    # Get sampling frequency
    sfreq = MNE.info['sfreq']

    # Extract time series from data
    TS, _ = extractTS(MNE = MNE, ch_list = ch_list, window = window, clst_method = 'append')

    # Initzialize result arrays
    LY = []
    # Error given by alghorithm, to use in case 'avg_trials == True'
    E_LY = []

    # Cycle around trials
    for ts in TS:
        # Cycle around clusters
        for t in ts:

            # Get mean period of the time series through power spectrum analysis
            # this method is not very robust to noise if the estimated period is too
            # large for the embeddin dimension
            if dt is None:
                f, ps = periodogram(t)
                dt = int(np.sum(ps)/np.sum(f*ps))
                if verbose == True:
                    print('Average period: ', dt)

            # Cycle around embeddings
            for m in embeddings:

                emb_ts = multi_embedding(c_ts = t, embedding = m, tau = tau)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                ly, e_ly, _, _, _ = lyap(dist_matrix = dist_matrix, dt = dt, w = w, sfreq = sfreq, T = emb_ts.shape[1], verbose = verbose)

                LY.append(ly)
                E_LY.append(e_ly)

    # Returns list in -C style ordering

    return LY, E_LY

# Iterable function generator
def it_lyapunov(info: dict, parameters: dict, cython = False):

    global iterable

    # Build Largest Lyapunov Exponent iterable function
    def iterable(MNE_l: list):

        LY, E_LY = lyapunov(MNE = MNE_l[0],
                            ch_list = info['ch_list'],
                            window = info['window'],
                            tau = parameters['tau'],
                            embeddings = parameters['embeddings'],
                            dt = parameters['dt'],
                            w = parameters['w'],
                            m_norm = parameters['m_norm'],
                            cython = cython, verbose = False)

        return LY, E_LY

    return iterable
