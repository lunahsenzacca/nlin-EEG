import mne

import numpy as np

# Autocorrelation time according to first minimum of Mutual Information (FMMI)
from teaspoon.parameter_selection.MI_delay import MI_for_delay

# Autocorrelation time according to first minimum of Mutual Information (FMMI)
from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau

from core import extractTS

# Delay time of channel time series of a specific trial
def delay_time(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list|tuple,
               tau_method: str, clst_method: str, window = [None,None]):

    # Initzialize result array
    tau = []

    TS, _ = extractTS(MNE = MNE, ch_list = ch_list, window = window, clst_method = clst_method)

    # Cycle around trials
    for ts in TS:
        # Cycle around clusters
        for t in ts:

            ctau = []

            # Cycle around electrodes
            for t_ in t:
                if tau_method == 'mutual_information':
                    ctau.append(MI_for_delay(t_))

                elif tau_method == 'autocorrelation':
                    ctau.append(autoCorrelation_tau(t_))

            if clst_method == 'mean':
                tau.append(np.asarray(ctau).mean())

            elif clst_method == 'append':
                tau.append(np.asarray(ctau).max())

    return tau

# Iterable function generator
def it_delay_time(info: dict, parameters: dict):

    global iterable

    # Build Correlation Sum iterable function
    def iterable(MNE: list):

        TAU = delay_time(MNE = MNE[0],
                         ch_list = info['ch_list'],
                         window = info['window'],
                         tau_method = parameters['tau_method'],
                         clst_method = parameters['clst_method'])

        return TAU

    return iterable
