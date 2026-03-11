import mne

import json

import numpy as np

from os.path import join

from scipy.spatial.distance import squareform

from core import extractTS, multi_embedding, distance_matrix, to_disk

# Recurrence Plot for a single distance matrix
def rec_plt(dist_matrix: np.ndarray, r: float, T: int):

    N = dist_matrix.shape[0]

    rplt = np.full((T,T), 2, dtype = np.int8)

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            # Get value of theta
            if dist_matrix[i,j] < r:
                
                rplt[i,j] = 1
                rplt[j,i] = 1

            else:

                rplt[i,j] = 0
                rplt[j,i] = 0

    return rplt

# Recurrence Plot of channel time series of a specific trial
def recurrence(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list|tuple,
               embeddings: list, tau: str|int, th_method: str, th_values: list,
               m_norm: bool = False, window: list = [None,None],
               cython: bool = False, memory_safe: bool = False, tmp_path: str | None  = None):

    if cython == True:

        from cython_modules import c_core

    # Prepare temporary file storage
    sv_file = ''
    if memory_safe is True and type(tmp_path) is str:

        id = MNE.info['description']

        sv_file = join(tmp_path, id + '.npz')

        np.savez_compressed(sv_file)


    # Extract time series from data
    TS, _ = extractTS(MNE = MNE, ch_list = ch_list, window = window, clst_method = 'append')

    # Initzialize result array
    RP = []

    # Cycle around trials
    for ts in TS:

        # Cycle around clusters
        for t in ts:

            # Cycle around embeddings
            for m in embeddings:
                emb_ts = multi_embedding(c_ts = t, embedding = m, tau = tau)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                if th_method == 'diameter':

                    rvals = [np.max(dist_matrix)*th_value for th_value in th_values]

                elif th_method == 'absolute':

                    rvals = th_values

                else:

                    raise ValueError('Unknown threshold method')

                for r in rvals:

                    if cython == False:

                        rp = rec_plt(dist_matrix = dist_matrix, r = r, T = t.shape[-1])

                    else:

                        rp = c_core.rec_plt(dist_matrix = dist_matrix, r = r, T = t.shape[-1])

                    rp = squareform(rp, checks = False)

                    if memory_safe is True and type(tmp_path) == str:

                        to_disk(arr = np.asarray(rp, dtype = np.int8), sv_file = sv_file)

                    else:

                        RP.append(rp)

    if memory_safe is True and type(tmp_path) == str:

        RP = sv_file

        with open(f'{tmp_path}/backup.json', 'r+' ) as f:

            d = json.load(f)

            d['succeded'].append(id)

            f.seek(0)

            json.dump(d, f, indent = 2)

    # Returns list in -C style ordering

    return RP

# Iterable function generator
def it_recurrence(info: dict, parameters: dict, cython = False, memory_safe = False, tmp_path = None):

    global iterable

    # Build persistance diagrams iterable function 
    def iterable(MNE_l: list):

        RP = recurrence(MNE = MNE_l[0],
                        ch_list = info['ch_list'],
                        window = info['window'],
                        tau = parameters['tau'],
                        embeddings = parameters['embeddings'],
                        m_norm = parameters['m_norm'],
                        th_method = parameters['th_method'],
                        th_values = parameters['th_values'],
                        cython = cython,
                        memory_safe = memory_safe,
                        tmp_path = tmp_path)

        return RP

    return iterable
