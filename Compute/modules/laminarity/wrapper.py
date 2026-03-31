# Usual suspects
import json

# Laminarity computation functions
from modules.laminarity import laminarity

# Multiprocessing wrappers
from parallelizer import stacked_calculator

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'laminarity'

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

### DATA PATHS ###

# Label of Recurrence Plot calculation
load_calc_lb = info['load_calc_lb']

# Label for parameter selection
calc_lb = parameters['calc_lb']

### PARAMETERS FOR LAMINARITY COMPUTATION ###

# Minimum lengths of diagonal lines considered
min_vlengths = parameters['min_vlengths']

add_info = {
    'calc_lb': calc_lb,

    'min_vlengths': min_vlengths,
}

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(parameters['embeddings']),len(parameters['th_values']),len(min_vlengths)]

# Script main method
if __name__ == '__main__':

    print('\n    LAMINARITY SCRIPT')

    RP = laminarity.get_recurrence(info = info, load_calc_lb = load_calc_lb)

    # Updated info dictionary
    info = info | add_info

    stacked_calculator(laminarity.it_laminarity(parameters = parameters), previous = RP,
                       info = info, fshape = fshape)

