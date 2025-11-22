import json

# Print data with wrapper
# Info dictionary about observable
with open('./.tmp/last.json', 'r') as f:

    info = json.load(f)

extra_instructions = {}

# Import the wrapper 
from plotting.wrappers import simple_plot

# Output images in code repos (Who needs notebooks anyway)
extra_instructions['save_here'] = False

# Set kitcat browser backend
extra_instructions['backend'] = 'kitcat'

# Set lower dpi and dimension multiplier for reasonable dimensions
extra_instructions['dpi'] = 200
extra_instructions['dim_m'] = 0.5

# Define main wrapper
def plot():

    # Launch the wrapper
    simple_plot(info = info, extra_instructions = extra_instructions, show = True, save = True, verbose = False)

    return

# Launch main wrapper
if __name__ == '__main__':

    plot()
