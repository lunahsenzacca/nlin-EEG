import matplotlib.cm as cm
import json

# Print data with wrapper
# Info dictionary about observable
with open('./.tmp/last.json', 'r') as f:

    info = json.load(f)

# Extra instructions dictionary for standard instructions override
extra_instructions = {
    'reduce_legend': [2,3,4,5,6,7,8,9],
    'avg': 'sub',
    'grid': (1,1),
    #'ylim': (-1e-5, 1e-5),
    #'legend_s': True,
    'alpha_m': 0.9,
    'colormap': cm.tab20c,
    'linewidth': 2,
    #'ylabel': 'ERPs $[V]$',
    }

#extra_instructions = None

# Import the wrapper 
from plotting import simple_plot

# Output images in code repos (Who needs notebooks anyway)
extra_instructions['save_here'] = False

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
