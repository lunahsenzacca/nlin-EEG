# Usual suspects
import json
import numpy as np

# Distances Distributions computation function
from modules.distances import distances

# Cython file compile wrapper
from core import cython_compile


# Multiprocessing wrappers
from parallelizer import loader, calculator

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'distances'

### CYTHON DEBUG PARAMETERS ###

# Cython implementation of the script
cython = False
cython_verbose = False

### MEMORY SAFE PARAMETER ###

# Whether to save results to disk to avoid RAM saturation
memory_safe = False
tmp_path = None

### SCRIPT PARAMETERS ###

### LOAD EXPERIMENT INFO AND SCRIPT PARAMETERS ###

with open('.tmp/info.json', 'r') as f:

    info = json.load(f)

with open(f'.tmp/modules/{obs_name}.json', 'r') as f:

    parameters = json.load(f)

### EXPERIMENT PARAMETERS ###

# Dataset name
exp_name = info['exp_name']

# Cluster label
clst_lb = info['clst_lb']

# Averaging method
avg_trials = info['avg_trials']

# Subject IDs
sub_list = info['sub_list']

# Conditions
conditions = info['conditions']

# Channels
ch_list = info['ch_list']

# Time window
window = info['window']

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clst = True
else:
    clst = False

### PARAMETERS FOR DISTANCES DISTRIBUTION COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Set different time delay for each time series
tau = parameters['tau']

# Embedding dimensions
embeddings = parameters['embeddings']

# Apply embedding normalization when computing distances
m_norm = parameters['m_norm']

# Distances for sampling the dependance
log_span = parameters['log_span']

edges = np.logspace(log_span[0], log_span[1], num = log_span[2], base = log_span[3])

parameters['edges'] = edges

# Updated info dictionary
info = {
        'obs_name' : obs_name,
        'calc_lb' : calc_lb,

        'tau' : tau,
        'embeddings' : embeddings,
        'm_norm' : m_norm,
        'log_span' : log_span,

        'clustered' : clst,
        'sub_list' : sub_list,
        'conditions' : conditions,
        'ch_list' : ch_list,
        'window' : window,

        'exp_name' : exp_name,
        'avg_trials': avg_trials,
        'clst_lb' : clst_lb
        }

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(edges)-1]

# Script main method
if __name__ == '__main__':

    print('\n    DISTANCES DISTRIBUTION SCRIPT')

    if cython == True:

        cython_compile(verbose = cython_verbose)

    if memory_safe == True:

        import os, tempfile

        os.makedirs('.tmp/memory_safe', exist_ok = True)

        tmp_dir = tempfile.TemporaryDirectory(prefix = f'{obs_name}-', dir = '.tmp/memory_safe')

        tmp_path = tmp_dir.name

    MNEs_iters = loader(info = info)

    calculator(distances.it_distances(info = info, parameters = parameters, cython = cython, memory_safe = memory_safe, tmp_path = tmp_path),
               MNEs_iters = MNEs_iters,
               info = info, fshapes = [fshape],
               dtypes = [np.int64],
               memory_safe = memory_safe, tmp_path = tmp_path)
