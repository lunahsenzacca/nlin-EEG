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
            t_birth, _, pairs = Persistence0D(t[0])
            nt_birth,_,npairs = Persistence0D(-t[0])

            parr  = np.zeros((4,max_pairs), dtype = np.float64)
            tarr  = np.zeros((2,max_pairs), dtype = np.int32)

            for i in range(0,max_pairs):
                if i < (min(len(pairs),len(npairs))):
                    parr[0:2,i] = pairs[-i-1]
                    parr[2:4,i] = npairs[-i-1]

                    tarr[0,i] = t_birth[-i-1]
                    tarr[1,i] = nt_birth[-i-1]
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
