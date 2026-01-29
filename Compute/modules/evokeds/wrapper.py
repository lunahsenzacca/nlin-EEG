# Usual suspects
import json

# Evokeds extraction functions
from modules.evokeds import evokeds

# Utility function for dimensional time and frequency domain of the experiment
from core import get_tinfo

# Multiprocessing wrappers
from parallelizer import loader, calculator

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

obs_name = 'evokeds'

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

# Load times for results array
_, times = get_tinfo(exp_name = exp_name, avg_trials = avg_trials, window = window)

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                't': list(times),

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(times)]

# Script main method
if __name__ == '__main__':

    print('\n    EVOKEDS PLOT SCRIPT')

    MNEs_iters, points = loader(info = info, with_std = True)

    calculator(evokeds.it_evokeds(info),
               MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = True)
