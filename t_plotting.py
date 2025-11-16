import matplotlib

import matplotlib.cm as cm
# Print data with wrapper

# Info dictionary about observable
info = {
    'exp_name': 'zbmasking_dense',
    'avg_trials': True,
    'obs_name': 'spectrum',
    'clust_lb': 'CFPO',
    'calc_lb': 'FL',
}

# Extra instructions dictionary for standard instructions override
extra_instructions = {
    'reduce_legend': [2,3,4,5,6,7,8,9],
    #'avg': 'sub',
    #'dim_m': 1.2,
    #'grid': (1,1),
    'ylim': (-6, 0),
    #'legend_s': True,
    'alpha_m': 0.9,
    'colormap': cm.tab20c,
    'linewidth': 1,
    #'ylabel': 'ERPs $[V]$',
    }

#extra_instructions = None

# Launch the wrapper
from plotting import simple_plot

# Output images in code repos (Who needs notebooks anyway)
extra_instructions['save_here'] = True

simple_plot(info = info, extra_instructions = extra_instructions, show = False, save = True, verbose = True)
