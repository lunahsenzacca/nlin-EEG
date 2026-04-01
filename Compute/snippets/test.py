from mne.stats import combine_adjacency, ttest_ind_no_p, permutation_cluster_test
from mne.channels import find_ch_adjacency

exp_lb = 'BM'
avg_trials = 'avg'
clst_lb = 'Global'
obs_name = 'rentropy'
calc_lb = 'MI_Multiple'

#exp_lb = 'BM'
#avg_trials = 'trl'
#clst_lb = 'TEST'
#obs_name = 'determinism'
#calc_lb = ''

idxs = [3,1,2]

vlim = (0,1)

RES, info = load(obs_name, f'{avg_trials}/{exp_lb}/{clst_lb}/{maind['obs_lb'][obs_name]}/{calc_lb}')

M = []

c = 0
for i in range(0,len(info['sub_list'])):

    c_list = [RES[f'arr_{c + k}'][:,:,:,*idxs] for k in range(0,len(info['conditions']))]

    M.append(c_list)

    c += len(info['conditions'])

if avg_trials == 'avg':

    M = np.asarray(M)
   
    M = np.squeeze(M)

    M = np.swapaxes(M, 0,1)

else:

    nM = []

    for c in range(0,len(info['conditions'])):
        
        c_trials = []
        for s in range(0,len(info['sub_list'])):

            [c_trials.append(t) for t in np.squeeze(M[s][c])]

        nM.append(c_trials)

    M = nM

ex_file = os.listdir(f'../Cargo/toMNE/avg/{exp_lb}')[0]
_ = mne.read_evokeds(f'../Cargo/toMNE/avg/{exp_lb}/{ex_file}', verbose = False)[0]

mne_info = _.info

del _, ex_file

def test(M, plot = False, p_th = 0.1):

    X = [M[i] for i in range(0,len(M))]

    adj, _ = find_ch_adjacency(info = mne_info, ch_type = 'eeg')

    adj = combine_adjacency(adj)

    threshold_tfce = dict(start=0, step=0.1)

    T, clst, clst_p, H0 = permutation_cluster_test(X = X,
                                                   n_permutations = 1000,
                                                   threshold = threshold_tfce,
                                                   stat_fun = ttest_ind_no_p,
                                                   adjacency = adj,
                                                   n_jobs = 10,
                                                   seed = 42)

    mask = np.where(clst_p <= p_th, True, False)

    if plot is True:

        fig, ax = plt.subplot_mosaic([['Un','Ma'],['widebar','widebar'],['t','p']], figsize = (9,9), height_ratios = [0.7,0.05,1])

        im = [
            plot_head(M[0].mean(axis = 0), exp_lb = 'BM', cmap = 'Oranges', axes = ax['Un'], vlim = vlim),
            plot_head(M[1].mean(axis = 0), exp_lb = 'BM', cmap = 'Oranges', axes = ax['Ma'], vlim = vlim),
            plot_head(T,exp_lb = 'BM', cmap = 'coolwarm', axes = ax['t']),
            plot_head([-np.log10(p) for p in clst_p],exp_lb = 'BM', cmap = 'GnBu', axes = ax['p'], mask = mask, mask_params = dict(markersize = 8))
        ]

        cbars = [
            plt.colorbar(mappable = im[0],
                         cax = ax['widebar'],
                         orientation = 'horizontal',
                         label = maind['obs_lb'][obs_name]),

            plt.colorbar(mappable = im[2],
                         ax = ax['t'],
                         orientation = 'horizontal',
                         label = 't-value'),

            plt.colorbar(mappable = im[3],
                         ax = ax['p'],
                         orientation = 'horizontal',
                         label = '$-log_{10}(p_{value})$')
        ]

        ax['Un'].set_title('Unmasked')
        ax['Ma'].set_title('Masked')

        ax['Un'].title.set_weight('bold')
        ax['Ma'].title.set_weight('bold')

        plt.show()

    rel_idx = np.argwhere(clst_p <= p_th).flatten()
    rel_names = [info['ch_list'][i] for i in rel_idx]

    if len(rel_names) != 0:
        print('Found the following significant electrodes:', *rel_names)
    else:
        print('No significant electrodes were found! :(')

    return T, clst, clst_p, H0

def plot_head(M: np.ndarray, exp_lb: str, axes = Axes, cmap: str = 'coolwarm', vlim: tuple = (None,None), **args):

    ex_file = os.listdir(f'../Cargo/toMNE/avg/{exp_lb}')[0]
    _ = mne.read_evokeds(f'../Cargo/toMNE/avg/{exp_lb}/{ex_file}', verbose = False)[0]

    info = _.info

    del _

    im, _ = mne.viz.plot_topomap(M, pos = info, axes = axes, show = False, cmap = cmap, sensors = 'k*', vlim = vlim, **args)

    return im

