from scipy.stats import ttest_ind
from mne.stats import combine_adjacency, permutation_t_test,ttest_ind_no_p, permutation_cluster_test
from mne.channels import find_ch_adjacency

exp_lb = 'BM'

RES, info = load('determinism', f'avg/{exp_lb}/Global/DET')

M = []

for i in range(0,len(info['sub_list'])):

    M.append([RES[f'arr_{2*i}'],RES[f'arr_{2*i + 1}']])

M = np.asarray(M)

M = M[:,:,0,0,:,2,1,2]

ex_file = os.listdir(f'../Cargo/toMNE/avg/{exp_lb}')[0]
_ = mne.read_evokeds(f'../Cargo/toMNE/avg/{exp_lb}/{ex_file}', verbose = False)[0]

mne_info = _.info

del _, ex_file

def test(M, plot = False):

    X = [M[:,i] for i in range(0,M.shape[1])]

    adj, _ = find_ch_adjacency(info = mne_info, ch_type = 'eeg')

    adj = combine_adjacency(adj)

    threshold_tfce = dict(start=0, step=0.1)

    def perm_t(a,b):

        return permutation_t_test(a-b, n_permutations = 50000)[0]

    T, clst, clst_p, H0 = permutation_cluster_test(X = X,
                                                   #n_permutations = 50000,
                                                   threshold = threshold_tfce,
                                                   stat_fun = perm_t,#ttest_ind_no_p,
                                                   adjacency = adj,
                                                   n_jobs = 10,
                                                   seed = 42)

    return T, clst, clst_p, H0
