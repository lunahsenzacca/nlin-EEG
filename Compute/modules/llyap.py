# Usual suspects
import json

# Cython file compile wrapper
from core import cython_compile

# Sub-wise function for Largest Lyapunov Exponent (LLE) computation
from core import lyapunov

# Multiprocessing wrappers
from parallelizer import loader, calculator

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

# Module name
obs_name = 'llyap'

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

### PARAMETERS FOR LARGEST LYAPUNOV EXPONENT COMPUTATION ###

# Label for parameter selection
calc_lb = parameters['calc_lb']

# Set different time delay for each time series
tau = parameters['tau']

# Embedding dimensions
embeddings = parameters['embeddings']

# Theiler window
w = parameters['w']

# Lenght of expansion phase
dt = parameters['dt']

# Apply embedding normalization when computing distances
m_norm = parameters['m_norm']

# Dictionary for computation variables
variables = {   
                'obs_name': obs_name,
                'calc_lb': calc_lb,

                'tau': tau,
                'embeddings': embeddings,
                'w': w,
                'dt': dt,
                'm_norm': m_norm,

                'clustered' : clst,
                'subjects' : sub_list,
                'conditions' : conditions,
                'pois' : ch_list,
                'window' : window
            }

### COMPUTATION ###

# Build Largest Lyapunov Exponent iterable function
def it_lyapunov(MNE_l: list):

    LY, E_LY = lyapunov(MNE = MNE_l[0], ch_list = ch_list,
                        embeddings = embeddings, tau = tau,
                        dt = dt, w = w, m_norm = m_norm,
                        cython= cython, window = window, verbose = False)

    return LY, E_LY

# Define shape of results
fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings)]

# Launch script with 'python -m llyap'
if __name__ == '__main__':

    print('\n    LARGEST LYAPUNOV EXPONENT SCRIPT')

    if cython == True:

        cython_compile(verbose = cython_verbose)

    MNEs_iters, points = loader(info = info)

    calculator(it_lyapunov, MNEs_iters = MNEs_iters, points = points,
               info = info, variables = variables, fshape = fshape,
               with_err = True)
