from scipy.spatial.distance import squareform

L = ['../Cargo/results','avg','BMD','Global','RP','Fixed(0_1)']

file_path = os.path.join(*L,'recurrence.npz')
info_path = os.path.join(*L,'info.json')

M = np.load(file_path)

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

def heat(file: str, idxs: list):

    trials = M[file]

    shape = trials.shape

    trials = np.sum(trials, axis = 0, dtype = np.float64)/shape[0]

    trials = trials.reshape([int(np.prod(shape[1:-1])),shape[-1]])

    heat = []

    cut = 0
    for i, t in enumerate(trials):

        h = squareform(t)

        if i == 0:
            for j in range(0,len(h)):
                h[j,j] = 1
                if h[j,0] == 2:
                    cut = j
                    break

        h = h[:cut,:cut]

        heat.append(h)

    square = heat[0].shape[-2:]

    heat = np.log(np.asarray(heat) + 0.001)

    heat = heat.reshape([*shape[1:-1],*square])

    plt.imshow(heat[*idxs], cmap = 'Oranges')
    plt.show()

    return
