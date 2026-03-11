#from snippets.setup import *
from parallelizer import mp_wrapper
from scipy.spatial.distance import squareform
from multiprocessing import Pool
from rich.progress import track

L = ['../Cargo/results','trl','BMD','Global','RP','Fixed']

file_path = os.path.join(*L,'recurrence.npz')
info_path = os.path.join(*L,'info.json')

M = np.load(file_path)

def square(m):

    m = squareform(m)

    cut = 0
    for i in range(0,len(m)):
        m[i,i] = 1
        if m[i,0] == 2:
            cut = i
            break

    return m[:cut,:cut]

def run_lengths_ones(x: np.ndarray) -> np.ndarray:
    '''Lengths of consecutive 1-runs in a 1D array.'''
    x01 = (x != 0).astype(np.int8)           # signed: important for diff
    d = np.diff(np.r_[0, x01, 0])
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    return ends - starts

def determinism(trial: np.ndarray, min_length: int = 3, exclude_main: bool = True):

    if trial.ndim == 1:
        trial = squareform(trial)

    n, m = trial.shape

    if n != m:
        raise ValueError("M must be square.")

    cut = 0
    for i in range(0,len(trial)):
        trial[i,i] = 1
        if trial[i,0] == 2:
            cut = i
            break

    trial = trial[:cut,:cut]

    num = 0  # points in diagonals with length >= min_length
    den = 0  # points in all diagonals (length >= 1)
 
    for offset in range(-(n - 1), n):
        if exclude_main and offset == 0:
            continue
        rl = run_lengths_ones(np.diagonal(trial, offset=offset))
        if rl.size == 0:
            continue
        den += rl.sum()
        num += rl[rl >= min_length].sum()

    if den == 0:
        return 0.0

    return num / den


DET = []
for id in track(M.files, description = '[red]Processing:', total = len(M.files)):

    flat = flat_recurrence(M, id)

    iters = [i for i in range(0,len(flat))]

    def it_determinism(idx: int):

        DET = determinism(flat[idx])

        return DET

    results = mp_wrapper(it_determinism, iters,
                         workers = 10,
                         chunksize = 20,
                         description = 'Trying:',
                         disable_bar = True)

    DET.append(results)

    del flat
