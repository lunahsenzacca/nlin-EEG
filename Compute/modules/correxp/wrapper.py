# Usual suspects
import json

import numpy as np

# Correlation Exponent computation functions
from modules.correxp import correxp

# Multiprocessing wrappers
from parallelizer import calculator

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'correxp'

### LOAD EXPERIMENT INFO AND SCRIPT PARAMETERS ###


### BETTER SORT OU INFO DICTIONARY OF CORRELATION SUM AND RESULT DICTIONARY
### THIS SHOULD LOAD THE CORRELATION SUM INFO OBJECT
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

# Label of Correlation Sum calculation
load_calc_lb = parameters['load_calc_lb']

# Label for parameter selection
calc_lb = parameters['calc_lb']

### PARAMETERS FOR FREQUENCY SPECTRUM COMPUTATION ###

# Number of points in moving average for derivative computation
n_points = parameters['n_points']

# Apply gaussian filter to results for smoothing
gauss_filter = parameters['gauss_filter']

# Parameters of the gaussian filter
scale = parameters['scale'] #0.01
cutoff = parameters['cutoff'] #5

add_variables = {
    'calc_lb': calc_lb,

    'n_points': n_points,
    'gauss_filter': gauss_filter,
    'scale': scale,
    'cutoff':cutoff
}

# Script main method
if __name__ == '__main__':

    print('\n    CORRELATION EXPONENT SCRIPT')

    log_CS_iters, points, variables = correxp.correxp_getcorrsum(info = info, load_calc_lb = load_calc_lb)

    embeddings = variables['embeddings']
    log_r = variables['log_r']

    # Check if mobile average leaves more than three cooridnates
    rlen = len(log_r) - n_points + 1
    if rlen < 4:
        raise ValueError('\n\'n_points\' too big, choose a smaller value')

    # Get reduced array for mobile average
    r_log_r = []
    for i in range(0,rlen):
        r_log_r.append(np.mean(np.asarray(log_r)[i:i + n_points]))

    # Create homogeneous array averaging across trial results
    fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(r_log_r)]

    variables = variables | add_variables

    calculator(correxp.it_correlation_exponent(variables),
               MNEs_iters = log_CS_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = True)


