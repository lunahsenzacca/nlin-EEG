# Usual suspects
import json

import numpy as np

# Correlation Sum computation functions
from modules.corrsum import corrsum

# Cython file compile wrapper
from core import cython_compile

# Multiprocessing wrappers
from parallelizer import loader, calculator

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'corrsum'

### CYTHON DEBUG PARAMETERS ###

# Cython implementation of the script
cython = False
cython_verbose = False

### LOAD EXPERIMENT INFO AND SCRIPT PARAMETERS ###

with open('./.tmp/info.json', 'r') as f:

    info = json.load(f)

with open(f'./.tmp/modules/{obs_name}.json', 'r') as f:

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

### PARAMETERS FOR CORRELATION SUM COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Embedding dimensions
embeddings = parameters['embeddings']

# Set different time delay for each time series
tau = parameters['tau']

# Theiler window
w = parameters['w']

# Apply embedding normalization when computing distances
m_norm = parameters['m_norm']

# Distances for sampling the dependance
log_span = parameters['log_span']

r = np.logspace(log_span[0], log_span[1], num = log_span[2], base = log_span[3])

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'tau' : tau,
                'w': w,
                'embeddings' : embeddings,
                'm_norm': m_norm,
                'log_span': log_span,
                'log_r': list(np.log(r)),

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r)]

# Script main method
if __name__ == '__main__':

    print('\n    CORRELATION SUM SCRIPT')

    if cython == True:

        cython_compile(verbose = cython_verbose)

    MNEs_iters, points = loader(info = info)

    calculator(corrsum.it_correlation_sum(info = info, parameters = parameters, cython = cython),
               MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = False)
