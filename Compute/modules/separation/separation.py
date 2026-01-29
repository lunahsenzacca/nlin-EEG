import mne

import numpy as np

from core import extractTS, multi_embedding, distance_matrix

# Spacetime Separation for a single distance matrix
def sep(dist_matrix: np.ndarray, percentiles: list, T: int):

    N = dist_matrix.shape[0]

    n = len(percentiles)

    splt = np.full((n,T), 0, dtype = np.float64)

    # Compose distribution of distances for each relative time distance
    for i in range(0,N):

        dist = []
        for j in range(0, N - i):

            dist.append(dist_matrix[j,i + j])

        if (N - i) > 10:

            perc = np.percentile(dist, percentiles)

            splt[:,i] = perc

    return splt

# Correlation sum of channel time series of a specific trial
def separation(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list|tuple,
                    embeddings: list, tau: str|int, percentiles: list,
                    m_norm = False, window = [None,None], cython = False):

    if cython == True:

        from cython_modules import c_core

    # Extract time series from data
    TS, _ = extractTS(MNE = MNE, ch_list = ch_list, window = window, clst_method = 'append')

    # Initzialize result array
    SP = []

    # Cycle around trials
    for ts in TS:
        # Cycle around clusters
        for t in ts:

            # Cycle around embeddings
            for m in embeddings:
                emb_ts = multi_embedding(c_ts = t, embedding = m, tau = tau)

                if cython == False:

                    dist_matrix = distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                    sp = sep(dist_matrix = dist_matrix, percentiles = percentiles, T = t.shape[-1])

                else:

                    dist_matrix = c_core.distance_matrix(emb_ts = emb_ts, m_norm = m_norm, m = np.sqrt(len(emb_ts)))

                    sp = c_core.sep(dist_matrix = dist_matrix, percentiles = percentiles, T = t.shape[-1])

                SP.append(sp)

     # Returns list in -C style ordering

    return SP

# Iterable generator function
def it_separation(info: dict, parameters: dict, cython = False):

    global iterable

    # Build Spacetime Separation Plot iterable function
    def iterable(MNE_l: list):

        SP = separation(MNE = MNE_l[0],
                        ch_list = info['ch_list'],
                        window = info['window'],
                        tau = parameters['tau'],
                        embeddings = parameters['embeddings'],
                        m_norm = parameters['m_norm'],
                        percentiles = parameters['percentiles'],
                        cython = cython)

        return SP

    return iterable
