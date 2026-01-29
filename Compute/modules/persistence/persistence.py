import mne

import numpy as np

# Zero Dimensional Sublevel Set Persistance Diagrams
from teaspoon.TDA.SLSP import Persistence0D

from core import extractTS

# Persistence of channel time series of a specific trial
def persistence(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list | tuple, max_pairs: int, window = [None,None]):

    TS, _ = extractTS(MNE = MNE, ch_list = ch_list, window = window, clst_method = 'mean')

    # Initzialize results array
    PS = []
    TPS = []

    # Loop around pois time series
    for ts in TS:
        for t in ts:

            # Get informations peristence of peak and valleys
            t_birth, _, pairs = Persistence0D(t)
            nt_birth,_,npairs = Persistence0D(-t)

            parr  = np.zeros((max_pairs,4), dtype = np.float64)
            tarr  = np.zeros((max_pairs,2), dtype = np.int32)

            for i in range(0,max_pairs):
                if i < (min(len(pairs),len(npairs))):
                    parr[i,0:2] = pairs[-i-1]
                    parr[i,2:4] = npairs[-i-1]

                    tarr[i,0] = t_birth[-i-1]
                    tarr[i,1] = nt_birth[-i-1]
                else:
                    break

            PS.append(parr)
            TPS.append(tarr)

    return PS, TPS

# Iterable function generator
def it_persistence(info: dict, parameters: dict):

    global iterable

    # Build persistance diagrams iterable function 
    def iterable(MNE_l: list):

        PS, TPS = persistence(MNE = MNE_l[0],
                              ch_list = info['ch_list'],
                              window = info['window'],
                              max_pairs = parameters['max_pairs'])

        return PS, TPS

    return iterable
