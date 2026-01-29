import mne

import numpy as np

from core import extractTS

# Frequency spectrum for a cluster time series
def spec(ts: np.ndarray, e_ts: np.ndarray, info: mne.Info, wf: float, N: None | int, n: None|int):

    ts = ts.mean(axis = 0)
    e_ts = e_ts.mean(axis = 0)

    psd, _ = mne.time_frequency.psd_array_welch(ts,
    sfreq = info['sfreq'],  # Sampling frequency from the evoked data
    fmin = info['highpass'], fmax = info['lowpass'],  # Focus on the filter range
    n_fft = int(len(ts)*wf),
    n_per_seg = int(len(ts)/wf),  # Length of FFT (controls frequency resolution)
    verbose = False)

    if type(n) is int and type(N) is int:

        psd_ = []
        for i in range(0,N):

            ts_r = np.random.normal(loc = ts, scale = e_ts*np.sqrt(n))

            psd_r, _ = mne.time_frequency.psd_array_welch(ts_r,
            sfreq = info['sfreq'],  # Sampling frequency from the evoked data
            fmin = info['highpass'], fmax = info['lowpass'],  # Focus on the filter range
            n_fft = int(len(ts)*wf),
            n_per_seg = int(len(ts)/wf),  # Length of FFT (controls frequency resolution)
            verbose = False)

            psd_.append(psd_r)

        e_psd = np.asarray(psd_).std(axis = 0)/np.sqrt(N)

    else:

        e_psd = np.zeros(len(psd))

    c_psd = psd.copy()

    psd = np.log10(c_psd)/10
    e_psd = e_psd/(np.log(10)*c_psd*10)

    del c_psd

    return psd, e_psd

# Frequency Spectrum for channel time series of a specific trial
def spectrum(MNE: mne.Evoked | mne.epochs.EpochsFIF, sMNE: None | mne.Evoked, ch_list: list | tuple, N: int, wf: float, window = [None,None]):

    # Extract time series from data
    TS, E_TS = extractTS(MNE = MNE, sMNE = sMNE, ch_list = ch_list, window = window, clst_method = 'mean')

    if type(MNE) is mne.Evoked:
        n = MNE.nave

    else:
        n = None

    # Initzialize result array
    SP = []
    E_SP = []

    # Cycle around trials
    for ts, e_ts in zip(TS, E_TS):
        # Cycle around clusters
        for t, e_t in zip(ts, e_ts):

            psd, e_psd = spec(ts = t, e_ts = e_t, info = MNE.info, wf = wf, N = N, n = n)

            SP.append(psd)
            E_SP.append(e_psd)

    return SP, E_SP

# Iterable function generator
def it_spectrum(info: dict, parameters: dict):

    global iterable

    # Build Spectrum Plotting iterable function
    def iterable(MNE_l: list):

        SP, E_SP = spectrum(MNE = MNE_l[0], sMNE = MNE_l[1],
                            ch_list = info['ch_list'],
                            window = info['window'],
                            N = parameters['N'],
                            wf = parameters['wf'])

        return SP, E_SP

    return iterable
