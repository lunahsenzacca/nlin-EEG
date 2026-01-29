# Usual suspects
import json

# Delay Time computation functions
from modules.delay import delay

# Multiprocessing wrappers
from parallelizer import loader, calculator

#Our Very Big Dictionary
from init import get_maind

maind = get_maind()

obs_name = 'delay'

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

### PARAMETERS FOR DELAY TIME COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Choose delay time computation method
tau_method = parameters['tau_method']

# Choose clustering method
clst_method = parameters['clst_method']

# Check if we are clustering electrodes
if type(ch_list) == tuple:
    clst = True
else:
    clst = False

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'tau_method': tau_method,
                'clst_method': clst_method,

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list)]

# Script main method
if __name__ == '__main__':

    print('\n    DELAY TIME SCRIPT')

    MNEs_iters, points = loader(info = info)

    calculator(delay.it_delay_time(info = info, parameters = parameters),
               MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = False)
