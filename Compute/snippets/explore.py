L = ['../Cargo/results','trl','BMD','Global','RP','Fixed']

file_path = os.path.join(*L,'recurrence.npz')
info_path = os.path.join(*L,'info.json')

#M = np.load(file_path)

with open(info_path, 'r') as f:
    info = json.load(f)

def show(file: str, idxs: list):

    m = squareform(M[file][*idxs])

    cut = 0
    for i in range(0,len(m)):
        m[i,i] = 1
        if m[i,0] == 2:
            cut = i
            break

    m = m[:cut,:cut]

    plt.imshow(m, cmap = 'Oranges')
    plt.show()

    del m

    return

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
