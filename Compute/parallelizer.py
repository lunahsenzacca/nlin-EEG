import json

import numpy as np

from core import loadMNE, save_results

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

    return flat

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
    MNEs_iters = flatMNEs(MNEs = MNEs)

    print('DONE!')

    return MNEs_iters

# Parallelized Observable computation
def calculator(it_observable, MNEs_iters: list,
               info: dict, fshape: list,
               with_err = False, dtype = np.float64,
               memory_safe = False, tmp_path = None,
               extra_res = False, extra_lb = None, extra_dtype = None):

    # IMPLEMENT A SMART WAY OF COMPUTING ETA

    if memory_safe is True:

        with open(f'{tmp_path}/backup.json', 'w') as f:
            json.dump({ 'succeded': [] }, f, indent = 2)

    print(f'\nComputing {maind['obs_nm'][info['obs_name']]}...\n')

    results_ = mp_wrapper(it_observable, iterable = MNEs_iters,
                          workers = workers,
                          chunksize = chunksize,
                          description = '[red]Processing:',
                          transient = False)

    # ALL OF THIS SHOULD BE MUCH MORE ELEGANT AT SOME POINT, AND GENERALIZED FOR ANY NUMBER OF OUTPUTS MAYBE
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

    if memory_safe is True:

        d = { 'results': results, 'e_results': e_results, 'fshape': fshape }

        with open(f'{tmp_path}/backup.json', 'w' ) as f:
            json.dump(d, f, indent = 2)

    if extra_res == False:

        save_results(results = results,
                     fshape = fshape,
                     info = info,
                     sv_name = info['obs_name'],
                     dtype = dtype,
                     e_results = e_results,
                     memory_safe = memory_safe)

    elif with_err == False and extra_res == True:

        save_results(results = results,
                     fshape = fshape,
                     info = info,
                     sv_name = info['obs_name'],
                     dtype = dtype,
                     memory_safe = memory_safe)

        save_results(results = e_results,
                     fshape = fshape,
                     info = info,
                     sv_name = f'{info['obs_name']}_{extra_lb}',
                     dtype = extra_dtype,
                     memory_safe = memory_safe)

    else:
        raise ValueError('Only error or extra value can be saved!')

    print('\nDONE!')

    # Save info file to .tmp for faster relaunching
    with open('.tmp/last.json','w') as f:
        json.dump(info, f, indent = 2)

    print('')

    return
