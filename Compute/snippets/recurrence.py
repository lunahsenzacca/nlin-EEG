from scipy.spatial.distance import squareform
from scipy.stats import ttest_ind
from seaborn import heatmap
from rich.progress import track

exp_lb = 'BM'
avg_trials = 'avg'
clst_lb = 'Global'
obs_name = 'recurrence'
calc_lb = 'Multiple'

#exp_lb = 'BM'
#avg_trials = 'trl'
#clst_lb = 'TEST'
#obs_name = 'determinism'
#calc_lb = ''

idxs = [0,0,0]

vlim = (0.2,0.8)

M, info = load(obs_name, f'{avg_trials}/{exp_lb}/{clst_lb}/{maind['obs_lb'][obs_name]}/{calc_lb}')

def peak(file: str, idxs: list, title: str = '', sv_name: str | None = None):

    m = squareform(M[file][*idxs])

    cut = len(m)
    for i in range(0,len(m)):
        m[i,i] = 1
        if m[i,0] == 2:
            cut = i
            break

    m = m[:cut,:cut]

    print(m.shape)


    fig, ax = plt.subplots(1,1, figsize = (5,5))

    heatmap(m, cmap = 'Oranges', cbar = False, ax = ax, xticklabels = 100, yticklabels = 100, square = True)

    ax.invert_yaxis()

    fig.suptitle(title)

    plt.tight_layout()

    plt.show()

    if type(sv_name) == str:

        plt.savefig(f'../Cargo/pics/{sv_name}.png', dpi = 300)

    plt.close()

    del m

    return

def heat(file: str, idxs: list | None = None, show = True, cut = True):

    trials = M[file]

    shape = trials.shape

    trials = np.sum(trials, axis = 0, dtype = np.float64)/shape[0]

    trials = trials.reshape([int(np.prod(shape[1:-1])),shape[-1]])

    heat = []

    c = 0
    for i, t in enumerate(trials):

        h = squareform(t)

        for j in range(0,len(h)):
            h[j,j] = 1

        heat.append(h)

    square = heat[0].shape[-2:]

    heat = np.asarray(heat)

    heat = heat.reshape([*shape[1:-1],*square])

    if type(idxs) == list:

        heat = heat[*idxs]

        if cut is True:
            for i in range(0,len(heat)):

                if heat[i,0] == 2:
                    heat = heat[:i,:i]
                    break

        if show is True:
            fig, ax = plt.subplots(1,1, figsize = (5,5))
            heatmap(heat, cmap = 'Oranges', vmax = 1, cbar = False, ax = ax, xticklabels = 100, yticklabels = 100, square = True)
            plt.show()

            return

    return heat

def confront(idxs: list | None = None, cut = False):

    confront_0 = heat('arr_0', idxs = idxs, show = False, cut = cut)
    confront_1 = heat('arr_1', idxs = idxs, show = False, cut = cut)

    for i in track(range(1,len(info['sub_list'])),
                   total = len(info['sub_list'])-1):

        confront_0 += heat(f'arr_{2*i}', idxs = idxs, show = False, cut = cut)
        confront_1 += heat(f'arr_{2*i+1}', idxs = idxs, show = False, cut = cut)

    confront = np.concatenate((confront_0[np.newaxis],confront_1[np.newaxis]), axis = 0)/len(info['sub_list'])

    return confront

def split():

    split_0 = heat('arr_0', show = False, cut = False)[np.newaxis]
    split_1 = heat('arr_1', show = False, cut = False)[np.newaxis]

    for i in range(1,len(info['sub_list'])):

        split_0 = np.concatenate((split_0,heat(f'arr_{2*i}', show = False, cut = False)[np.newaxis]), axis = 0)
        split_1 = np.concatenate((split_1,heat(f'arr_{2*i+1}', show = False, cut = False)[np.newaxis]), axis = 0)

    split = np.concatenate((split_0[:,np.newaxis],split_1[:,np.newaxis]), axis = 1)

    return split

def show(idxs: list | None = None):

    if idxs is None:

        conf = confront()

        cc = conf[0] - conf[1]

        for j, emb in enumerate(info['embeddings']):
            for i, poi in enumerate(info['ch_list']):

                print(f'\nVVV Showing POI: {poi}, m: {emb}, th: {info['th_values']} VVV\n')

                fig, ax = plt.subplots(1,3, figsize = (8,4), gridspec_kw={'width_ratios': [1, 1, 0.2]})

                heatmap(cc[0,i,j,0], center = 0, cmap = 'coolwarm', cbar = False, ax = ax[0], xticklabels = 100, yticklabels = 100, square = True)
                heatmap(cc[0,i,j,1], center = 0, cmap = 'coolwarm', cbar = True, ax = ax[1], xticklabels = 100, yticklabels = 100, square = True, cbar_ax = ax[2])

                ax[0].invert_yaxis()
                ax[1].invert_yaxis()

                ax[2].set(box_aspect = 20, visible = True)

                plt.tight_layout()

                plt.show()

                plt.close()

        return

    else:

        conf = confront(idxs = idxs, cut = True)

        fig, ax = plt.subplots(1,2, figsize = (5,4), gridspec_kw={'width_ratios': [1, 0.2]})

        cc = conf[0] - conf[1]

        heatmap(cc, center = 0, cmap = 'coolwarm', cbar = True, ax = ax[0], cbar_ax = ax[1], xticklabels = 100, yticklabels = 100, square = True)

        ax[0].invert_yaxis()

        ax[1].set(box_aspect = 20, visible = True)

        plt.show()

        return fig, ax

def ttest():

    spl = split()

    t = ttest_ind(spl[:,0],spl[:,1])

    cc = t.statistic

    for j, emb in enumerate(info['embeddings']):
        for i, poi in enumerate(info['ch_list']):

            print(f'\nVVV Showing POI: {poi}, m: {emb}, th: {info['th_values']} VVV\n')

            fig, ax = plt.subplots(1,3, figsize = (8,4), gridspec_kw={'width_ratios': [1, 1, 0.2]})

            heatmap(cc[0,i,j,0], center = 0, cmap = 'coolwarm', cbar = False, ax = ax[0], xticklabels = 100, yticklabels = 100, square = True)
            heatmap(cc[0,i,j,1], center = 0, cmap = 'coolwarm', cbar = True, ax = ax[1], xticklabels = 100, yticklabels = 100, square = True, cbar_ax = ax[2])

            ax[0].invert_yaxis()
            ax[1].invert_yaxis()

            ax[2].set(box_aspect = 20, visible = True)

            plt.tight_layout()

            plt.show()

            plt.close()

    return
