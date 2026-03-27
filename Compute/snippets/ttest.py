from scipy.stats import ttest_ind

RES, info = load('determinism', 'avg/BM/Global/DET')

M = []

for i in range(0,len('sub_list')):

    M.append([RES[f'arr_{2*i}'],RES[f'arr_{2*i + 1}']])

M = np.asarray(M)


