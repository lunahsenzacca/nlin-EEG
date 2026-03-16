from scipy.spatial.distance import squareform
from seaborn import heatmap

L = ['../Cargo/results','avg','BMD','C Frontal-Occipital','RP','Multiple']

file_path = os.path.join(*L,'recurrence.npz')
info_path = os.path.join(*L,'info.json')

M = np.load(file_path)

with open(info_path, 'r') as f:
    info = json.load(f)

def peak(file: str, idxs: list):

    m = squareform(M[file][*idxs])

    cut = len(m)
    for i in range(0,len(m)):
        m[i,i] = 1
        if m[i,0] == 2:
            cut = i
            break

    m = m[:cut,:cut]

    print(m.shape)

    plt.imshow(m, cmap = 'Oranges')
    plt.show()

    del m

    return

def heat(file: str, idxs: list | None = None, show = True, cut = True):

    trials = M[file]

    shape = trials.shape

    trials = np.sum(trials, axis = 0, dtype = np.float64)/shape[0]

    trials = trials.reshape([int(np.prod(shape[1:-1])),shape[-1]])

    heat = []

    cut = 0
    for i, t in enumerate(trials):

        h = squareform(t)

        if i == 0:
            cut = len(h)
            for j in range(0,len(h)):
                h[j,j] = 1
                if h[j,0] == 2:
                    cut = j
                    break

        if cut:
            h = h[:cut,:cut]

        heat.append(h)

    square = heat[0].shape[-2:]

    heat = np.asarray(heat)

    heat = heat.reshape([*shape[1:-1],*square])

    if show and type(idxs) == list:
        plt.imshow(heat[*idxs], cmap = 'Oranges')
        plt.show()

        return

    return heat

def confront():

    confront_0 = heat('arr_0', show = False, cut = False)
    confront_1 = heat('arr_1', show = False, cut = False)

    for i in range(1,len(info['sub_list'])):

        confront_0 += heat(f'arr_{2*i}', show = False, cut = False)
        confront_1 += heat(f'arr_{2*i+1}', show = False, cut = False)

    confront = np.concatenate((confront_0[np.newaxis],confront_1[np.newaxis]), axis = 0)/len(info['sub_list'])

    return confront

def show():

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
