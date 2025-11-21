import matplotlib.cm as cm

# Print data with wrapper
# Info dictionary about observable
info = {
    'exp_name': 'bmasking_dense',
    'avg_trials': True,
    'obs_name': 'spectrum',
    'clst_lb': 'FT',
    'calc_lb': None,
}

# Extra instructions dictionary for standard instructions override
extra_instructions = {
    'reduce_multi': [i for i in range(0,21)],
    #'avg': 'sub',
    #'grid': (1,1),
    #'ylim': (-1e-5, 1e-5),
    #'legend_s': True,
    #'alpha_m': 0.9,
    #'colormap': cm.tab20c,
    #'linewidth': 2,
    #'ylabel': 'ERPs $[V]$',
    }

#extra_instructions = None

# Import the wrapper 
from plotting import simple_plot

# Output images in code repos (Who needs notebooks anyway)
extra_instructions['save_here'] = True

# Set kitcat browser backend
extra_instructions['backend'] = 'kitcat'

# Set lower dpi and dimension multiplier for reasonable dimensions
extra_instructions['dpi'] = 200
extra_instructions['dim_m'] = 0.6

# Define main wrapper
def plot():

    # Launch the wrapper
    simple_plot(info = info, extra_instructions = extra_instructions, show = True, save = True, verbose = True)

    return

# Launch main wrapper
if __name__ == '__main__':

    plot()
