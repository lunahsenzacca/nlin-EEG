# Usual suspects
import json

# Spacetime Separation computation function
from modules.separation import separation

# Cython file compile wrapper
from core import cython_compile

# Utility function for dimensional time and frequency domain of the experiment
from core import get_tinfo

# Multiprocessing wrappers
from parallelizer import loader, calculator

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'separation'

### CYTHON DEBUG PARAMETERS ###

# Cython implementation of the script
cython = False
cython_verbose = False

### SCRIPT PARAMETERS ###

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

# Dataset name
exp_name = 'bmasking_dense'

# Cluster label
clust_lb = 'CFPO'

# Calcultation parameters label
calc_lb = 'F'

# Get data averaged across trials
avg_trials = True

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clst = True
else:
    clst = False

# Load times for results array
_, times = get_tinfo(exp_name = exp_name, avg_trials = avg_trials, window = window)

### PARAMETERS FOR SEPARATION PLOT COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Set different time delay for each time series
tau = parameters['tau']

# Embedding dimensions
embeddings = parameters['embeddings']

# Apply embedding normalization when computing distances
m_norm = parameters['m_norm']

# Set desired percentiles for isolines
percentiles = parameters['percentiles']

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'tau' : tau,
                'embeddings': embeddings,
                'm_norm': m_norm,
                'percentiles' : percentiles,
                'dt': list(times),

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(percentiles),len(times)]

# Script main method
if __name__ == '__main__':

    print('\n    SPACETIME SEPARATION SCRIPT')

    if cython == True:

        cython_compile(verbose = cython_verbose)

    MNEs_iters, points = loader(info = info)

    calculator(separation.it_separation(info = info, parameters = parameters, cython = cython),
               MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = False)
