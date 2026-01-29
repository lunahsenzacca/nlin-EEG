# Usual suspects
import json

import numpy as np

# Persistence computation functions
from modules.persistence import persistence

# Multiprocessing wrappers
from parallelizer import loader, calculator

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'persistence'

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

### PARAMETERS FOR PERSISTENCE COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Maximum number of persistance feature returned
max_pairs = parameters['max_pairs']

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'max_pairs': max_pairs,

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),max_pairs,2]

# Script main method
if __name__ == '__main__':

    print('\n    PERSISTENCE SCRIPT')

    MNEs_iters, points = loader(info = info)

    calculator(persistence.it_persistence(info = info, parameters = parameters),
               MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = False,
               extra_res = True, extra_lb = 'times', extra_dtype = np.int32)
