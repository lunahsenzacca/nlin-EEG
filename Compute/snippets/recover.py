
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
_, times = c.get_tinfo(exp_name = exp_name, avg_trials = avg_trials, window = window)

# Set threshold values
th_values = info['th_values']

embeddings = info['embeddings']

fshape = [len(sub_list),len(conditions),len(ch_list),len(embeddings),len(th_values),int(len(times)*(len(times) - 1)/2)]

results = ['/home/lunis/nlin-EEG/Compute/.tmp/memory_safe/recurrence-nrg3usui/' + i for i in os.listdir('.tmp/memory_safe/recurrence-nrg3usui')]

print(results,len(results))

c.save_results(results = results, fshape = fshape, info = info, sv_name = info['obs_name'], dtype = np.int8, memory_safe = True)
