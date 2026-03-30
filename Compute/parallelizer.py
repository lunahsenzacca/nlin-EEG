import json

import numpy as np

from core import loadMNE, save_results, flat_results

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
def mp_wrapper(function, iterable: list, workers: int, chunksize: int = chunksize, description: str = 'Be patient:', transient: bool = True, disable_bar: bool = False):

    with Pool(workers) as p:

        results = list(track(p.imap(function, iterable, chunksize = chunksize),
                       description = description,
                       total = len(iterable),
                       transient = transient,
                       disable = disable_bar))

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

# Parallelized Observable computation based on already saved results
def stacked_calculator(observable, previous: np.lib.npyio.NpzFile,
                       info: dict,
                       fshape: list,
                       res_type = 1, dtype = np.float64,
                       cut: int | None = -1):

    print(f'\nComputing {maind['obs_nm'][info['obs_name']]}...\n')

    results_ = []

    for id in track(previous.files, description = '[red]Processing:', total = len(previous.files), transient = False):

        flat, _ = flat_results(file = previous, id = id)

        iters = [i for i in range(0,len(flat))]

        global it_observable

        def it_observable(idx: int):

            RES = observable(flat[idx])

            return RES

        partial = mp_wrapper(it_observable,iters,
                             workers = workers,
                             chunksize = (len(iters)//100 + 1),
                             disable_bar = True)

        partial = np.asarray(partial).swapaxes(0,1)

        results_.append(partial)

    if res_type == 1:

        save_results(results = [results[0] for results in results_],
                     fshape = fshape,
                     cut = cut,
                     info = info,
                     sv_name = info['obs_name'],
                     dtype = dtype,
                     e_results = None)

    elif res_type == 2:

        save_results(results = [results[0] for results in results_],
                     fshape = fshape,
                     cut = cut,
                     info = info,
                     sv_name = info['obs_name'],
                     dtype = dtype,
                     e_results = [results[1] for results in results_])

    print('\nDONE!')

    # Save info file to .tmp for faster relaunching
    #with open('.tmp/last.json','w') as f:
    #    json.dump(info, f, indent = 2)

    print('')

    return

# Parallelized Observable computation
def calculator(it_observable, MNEs_iters: list,
               info: dict, fshapes: list[list],
               res_types = [1], dtypes = [np.float64],
               labels = [''], cuts = [-1],
               memory_safe = False, tmp_path = None):

    if len(fshapes) != len(res_types):

        raise ValueError('Unconsistent number of results, saving results will fail. Please check provided shapes and res types.')

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

    if memory_safe is True:

        d = { 'results': results_, 'fshapes': fshapes }

        with open(f'{tmp_path}/backup.json', 'w' ) as f:
            json.dump(d, f, indent = 2)

    c = 0
    for i in range(0,len(fshapes)):

        if res_types[i] == 1:

            save_results(results = [results[c] for results in results_],
                         fshape = fshapes[i],
                         cut = cuts[i],
                         info = info,
                         sv_name = info['obs_name'] + labels[c],
                         dtype = dtypes[i],
                         e_results = None,
                         memory_safe = memory_safe)

            c += 1

        elif res_types[i] == 2:

            save_results(results = [results[c] for results in results_],
                         fshape = fshapes[i],
                         cut = cuts[i],
                         info = info,
                         sv_name = info['obs_name'] + labels[c],
                         dtype = dtypes[i],
                         e_results = [results[c+1] for results in results_],
                         memory_safe = memory_safe)

            c += 2

    print('\nDONE!')

    # Save info file to .tmp for faster relaunching
    with open('.tmp/last.json','w') as f:
        json.dump(info, f, indent = 2)

    print('')

    return
