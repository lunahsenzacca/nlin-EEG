import mne

import json

import numpy as np

from os.path import join

from scipy.spatial.distance import squareform

from core import extractTS, multi_embedding, distance_matrix, to_disk

# Recurrence Plot for a single distance matrix
def binned_dist(dist_matrix: np.ndarray, edges: np.ndarray) -> np.ndarray:

    dists, _ = np.histogram(squareform(dist_matrix), bins = edges)

    return dists

# Recurrence Plot of channel time series of a specific trial
def distances(MNE: mne.Evoked | mne.epochs.EpochsFIF, ch_list: list|tuple,
              embeddings: list, tau: str|int, edges: np.ndarray,
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
    DS = []

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

                ds = binned_dist(dist_matrix = dist_matrix, edges = edges)

                if memory_safe is True and type(tmp_path) == str:

                    to_disk(arr = np.asarray(ds, dtype = np.int8), sv_file = sv_file)

                else:

                    DS.append(ds)

    if memory_safe is True and type(tmp_path) == str:

        ds = sv_file

        with open(f'{tmp_path}/backup.json', 'r+' ) as f:

            d = json.load(f)

            d['succeded'].append(id)

            f.seek(0)

            json.dump(d, f, indent = 2)

    # Returns list in -C style ordering

    return DS

# Iterable function generator
def it_distances(info: dict, parameters: dict, cython = False, memory_safe = False, tmp_path = None):

    global iterable

    # Build persistance diagrams iterable function 
    def iterable(MNE_l: list):

        DS = distances(MNE = MNE_l[0],
                        ch_list = info['ch_list'],
                        window = info['window'],
                        tau = parameters['tau'],
                        embeddings = parameters['embeddings'],
                        m_norm = parameters['m_norm'],
                        edges = parameters['edges'],
                        cython = cython,
                        memory_safe = memory_safe,
                        tmp_path = tmp_path)

        return [DS]

    return iterable
