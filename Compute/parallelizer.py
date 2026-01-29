import os
import json

import numpy as np

from core import loadMNE, collapse_trials, obs_path

from tqdm import tqdm

from rich.progress import track

from multiprocessing import Pool

from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 10
chunksize = 1

### MULTIPROCESSING WRAPPERS ###

# Create one dimensional list with list of MNE objects per subject per condition
def flatMNEs(MNEs: list):

    # Flatten MNEs nested list
    flat = [x for xs in MNEs for x in xs]

    # Save separation coordinates for 'collapse_trials' functions
    points = [[1 for j in range(0,len(MNEs[i]))] for i in range(0,len(MNEs))]

    return flat, points

# Multiprocessing helper function
def mp_wrapper(function, iterable: list, workers: int, description: str, transient = True, chunksize = chunksize):

    with Pool(workers) as p:

        results = list(track(p.imap(function, iterable, chunksize = chunksize),
                       description = description,
                       total = len(iterable),
                       transient = transient))

    return results

# Parallelized subject data loading
def loader(info: dict, with_std = False):

    global it_loadMNE

    def it_loadMNE(subID: str):

        MNEs = loadMNE(subID = subID, exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'], conditions = info['conditions'],
                       with_std = with_std)

        return MNEs

    print('\nReading subject data...\n')

    MNEs = mp_wrapper(function = it_loadMNE, iterable = info['sub_list'],
                      workers = workers,
                      chunksize = chunksize,
                      description = '[red]Loading data:')

    # Create flat iterable list of evokeds images
    MNEs_iters, points = flatMNEs(MNEs = MNEs)

    print('DONE!')

    return MNEs_iters, points

# Parallelized Observable computation
def calculator(it_observable, MNEs_iters: list, points: list,
               info: dict, variables: dict, fshape: list,
               with_err = False, dtype = np.float64,
               extra_res = False, extra_lb = None, extra_dtype = None):

    # IMPLEMENT A SMARTER WAY OF COMPUTING ETA

    #if window[0] is None:
    #    i_window = maind[exp_name]['window']
    #else:
    #    i_window = window

    ## Get absolute complexity of the script and estimated completion time
    #complexity = np.sum(np.asarray(points))*len(ch_list)*len(embeddings)*np.log(len(r))*(((maind[exp_name]['T'])**2)*(i_window[1]-i_window[0])**2)

    #velocity = 26e-7

    #eta = str(timedelta(seconds = int(complexity*velocity/workers)))

    print(f'\nComputing {maind['obs_nm'][info['obs_name']]}...\n')
    #print('\nNumber of single computations: ' + str(int(complexity)))
    #print('\nEstimated completion time < ~' + eta)
    #print('\nSpawning ' + str(workers) + ' processes...')

    results_ = mp_wrapper(it_observable, iterable = MNEs_iters,
                          workers = workers,
                          chunksize = chunksize,
                          description = '[red]Processing:',
                          transient = False)

    if with_err == True or extra_res == True:

        # Get separate results lists
        results = []
        e_results = []
        for r in results_:

            results.append(r[0])
            e_results.append(r[1])

    else:

        results = results_
        e_results = None

    if with_err == False:

        R = collapse_trials(results = results, points = points, fshape = fshape, dtype = dtype, e_results = e_results)

    elif with_err == False and extra_res == True:

        R = collapse_trials(results = results, points = points, fshape = fshape, dtype = dtype, e_results = None)
        R1 = collapse_trials(results = e_results, points = points, fshape = fshape, dtype = extra_dtype, e_results = None)

    else:
        raise ValueError('Only error or extra value can be saved!')

    print('\nDONE!')

    # Save results to local

    sv_path = obs_path(exp_name = info['exp_name'], obs_name = info['obs_name'], avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = info['calc_lb'])

    os.makedirs(sv_path, exist_ok = True)

    np.savez(sv_path + f'{info['obs_name']}.npz', *R)

    if extra_res == True:

        np.savez(sv_path + f'{info['obs_name']}_{extra_lb}.npz', *R)

    with open(sv_path + 'variables.json','w') as f:
        json.dump(variables, f, indent = 2)

    with open(sv_path + 'info.json','w') as f:
        json.dump(info, f, indent = 2)

    print('\nResults common shape: ', R[0].shape[1:])

    if info['avg_trials'] == False:

        print('\nTrials\n')

        for c, prod in enumerate([i + '_' + j for i in info['sub_list'] for j in info['conditions']]):
            print(f'{prod}: ', R[c].shape[0])

    print('')

    return
