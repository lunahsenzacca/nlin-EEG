import mne

import numpy as np

from core import extractTS, multi_embedding, distance_matrix

# Correlation Sum from a distance matrix
def corr_sum(dist_matrix: np.ndarray, r: float, w: int):

    N = dist_matrix.shape[0]

    if w == 0:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if dist_matrix[i,j] < r:

                    c += 1

        csum = (2/(N*(N-1)))*c

    else:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if dist_matrix[i,j] < r and (i - j) > w:

                    c += 1

        csum = (2/((N-w)*(N-w-1)))*c

    return csum

# Correlation sum of channel time series of a specific trial
def correlation_sum(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list|tuple,
                    embeddings: list, tau: str|int, w: int, rvals: list, 
                    m_norm = False, window = [None,None], cython = False):

    if cython == True:

        from cython_modules import c_core

    # Extract time series from data
    TS, _ = extractTS(MNE = MNE, ch_list = ch_list, window = window, clst_method = 'append')

    # Initzialize result array
    CS = []

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

                # Cycle around r values
                for r in rvals:

                    if cython == False:

                        cs = corr_sum(dist_matrix = dist_matrix, r = r, w = w)

                    else:

                        cs = c_core.corr_sum(dist_matrix = dist_matrix, r = r, w = w)

                    CS.append(cs)

    # Returns list in -C style ordering

    return CS

# Iterable function generator
def it_correlation_sum(info: dict, parameters: dict, cython = False):

    global iterable

    # Build Correlation Sum iterable function
    def iterable(MNE_l: list):

        CS = correlation_sum(MNE = MNE_l[0],
                             ch_list = info['ch_list'],
                             window = info['window'],
                             tau = parameters['tau'],
                             w = parameters['w'],
                             embeddings = parameters['embeddings'],
                             m_norm = parameters['m_norm'],
                             rvals = parameters['r'],
                             cython = cython)

        return CS

    return iterable
