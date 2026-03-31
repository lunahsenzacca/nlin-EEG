import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# import nilearn as nil
# import nibabel as nib
# from nilearn.connectome import ConnectivityMeasure
import mne
import pickle
from teaspoon.parameter_selection.MI_delay import MI_for_delay
import warnings
from scipy.stats import linregress
from teaspoon.parameter_selection.FNN_n import FNN_n
import seaborn as sns
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from scipy.signal import periodogram
import os

### TO TAKE ARGUMENTS ###

import argparse
from pathlib import Path

data_dir = Path('/external_data')

### Params ###

n_jobs = -1
print(n_jobs)
best_mean_tau = 19 # Start
n_timepoints = 962 # Start
best_mean_tau =  19 # AroundResp
HARDCORE_PARALLEL = True
HARDCORE_PARALLEL_EMB_RP = False

### Utils ###

# Time-delay embedding of a single time series
def td_embedding(ts: np.ndarray, embedding: int, tau: int):

    min_len = (embedding - 1)*tau + 1

    # Check if embedding is possible
    if len(ts) < min_len:

        print('Data lenght is insufficient, try smaller parameters')
        return
    
    # Set lenght of embedding
    m = len(ts) - min_len + 1

    # Get indexes
    idxs = np.repeat([np.arange(embedding)*tau], m, axis = 0)
    idxs += np.arange(m).reshape((m, 1))

    emb_ts = ts[idxs]

    emb_ts = np.asarray(emb_ts, dtype = np.float64)

    return emb_ts

# Faster distance matrix
def distance_matrix_scipy(emb_ts: np.ndarray, m_norm: bool = False, m: float | None = None) -> np.ndarray:
    X = np.asarray(emb_ts, dtype=np.float64).T  # (N, d)
    D = squareform(pdist(X, metric="euclidean"))
    if m_norm and (m is not None):
        D /= m
    return D

# Largest Lyapunov exponent for a single embedded time series [Rosenstein et al.]
def lyap(dist_matrix: np.ndarray, dt: int, w: int, sfreq: int, T: int, verbose = False):

    ds = dist_matrix

    # Construct separations trajectories with embedded data
    lnd = []
    for i, el in enumerate(ds):

        if i < T - w:

            # Select only distant points on the trajectory but not too far on the end
            jt = []
            for j in range(0,T):
                if abs(j-i) > dt and j < T - w:
                    jt.append(j)

            if len(jt) != 0:
                # Get nearest neighbour
                d0 = np.min(el[jt])
                js = int(np.argwhere(el == d0)[0][0])

            # Construct separation data
            for delta in range(0,w):
                if len(jt) != 0:
                    lnd.append(np.log(ds[i +delta,js + delta]))
                else:
                    lnd.append(np.nan)

    # Reshape results
    lnd = np.asarray(lnd).reshape((T - w,w))

    # Get slope for largest Lyapunov exponent
    x = np.asarray([i for i in range (0,w)])

    with warnings.catch_warnings():
        if verbose == False:
            warnings.simplefilter('ignore')
            y = np.nanmean(lnd, axis = 0)*sfreq
        else:
            y = np.nanmean(lnd, axis = 0)*sfreq

    fit = linregress(x,y)

    lyap = fit.slope
    lyap_e = fit.stderr

    return lyap, lyap_e, x, y, fit

# 1-D function Adymensional Gaussian kernel convolution
def gauss_kernel(function: np.ndarray, x: np.ndarray, scale: float, cutoff: int, order: int):

    i_len = len(function)

    # Create convoluted array
    c_function = np.copy(function)
    c_x = np.copy(x)

    # Create redundant head and tails
    f_tail = np.array([function[0] for i in range(0,cutoff)])
    f_head = np.array([function[-1] for i in range(0,cutoff)])

    x_tail = np.array([x[0]  for i in range(0,cutoff)])
    x_head = np.array([x[-1] for i in range(0,cutoff)])

    function = np.concatenate((f_tail, function, f_head))
    x = np.concatenate((x_tail, x, x_head))

    for i in range(0,i_len):

        vals = np.array([function[j] for j in range(i, 2*cutoff + i + 1)])
        if order == 0:
            ker = np.array([np.exp((x[j]-c_x[i])**2)/(2*(scale**2)) for j in range(i, 2*cutoff + i + 1)])
        elif order == 1:
            print('Order 1 kernel Not yet implemented')
            return

        c_function[i] = np.sum(vals*ker)/np.sum(ker)

    return c_function

# Correlation Sum from a Recurrence Plot
def corr_sum(dist_matrix: np.ndarray, r: float, w: int):

    N = dist_matrix.shape[0]

    if w == 0:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if dist_matrix[i,j] < r:

                    c += 1

        csum = (2/(N*(N-1)))*c

    else:

        c = 0
        for i in range(0,N):
            for j in range(0,i):

                if dist_matrix[i,j] < r and (i - j) > w:

                    c += 1

        csum = (2/((N-w)*(N-w-1)))*c

    return csum

# Correlation Exponent computation
def corr_exp(log_csum: list, log_r: list, n_points: int, gauss_filter: bool, scale = None, cutoff = None):

    rlen = len(log_r) - n_points + 1

    ce =  []
    e_ce = []
    n_log_r = []
    for i in range(0,rlen):

        m = np.array([(log_csum[i+j+1] - log_csum[i+j])/(log_r[i+j+1] - log_r[i+j]) for j in range(0,n_points-1)])

        # Get value for error of slope
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = linregress(x = np.asarray(log_r)[i:i + n_points], y = log_csum[i:i + n_points])

        ce.append(np.asarray(m).mean())
        e_ce.append(results.stderr)

        n_log_r.append(np.mean(np.asarray(log_r)[i:i + n_points]))

    ce = np.asarray(ce)
    e_ce = np.asarray(e_ce)

    n_log_r = np.asarray(n_log_r)

    # Apply Gaussian filtering for smoothing
    if gauss_filter is True:
        if cutoff or scale is None:
            raise ValueError('\'cutoff\' and \'scale\' need to be set for kernel computation')
        else:
            ce = gauss_kernel(function = ce, x = n_log_r, scale = scale, cutoff = cutoff, order = 0)
            e_ce = gauss_kernel(function = e_ce, x = n_log_r, scale = scale, cutoff = cutoff, order = 0)

    return  ce, e_ce

# Spacetime Separation Plot for a single embeddend time series
def sep_plt(dist_matrix: np.ndarray, percentiles: np.ndarray, T: int):

    N = dist_matrix.shape[0]

    n = percentiles.shape[0]

    splt = np.full((n,T), 0, dtype = np.float64)

    # Compose distribution of distances for each relative time distance
    for i in range(0,N):

        dist = []
        for j in range(0, N - i):

            dist.append(dist_matrix[j,i + j])

        if (N - i) > 2*n:

            perc = np.percentile(dist, percentiles)

            splt[:,i] = perc

    return splt

# Recurrence Plot for a single embeddend time series
def rec_plt(dist_matrix: np.ndarray, r: float, T: int):

    N = dist_matrix.shape[0]

    rplt = np.full((T,T), 0, dtype = np.int8)

    # Cycle through all different couples of points
    for i in range(0,N):
        for j in range(0,i + 1):

            # Get value of theta
            if dist_matrix[i,j] < r:
                
                rplt[i,j] = 1
                rplt[j,i] = 1
    np.fill_diagonal(rplt, 0)

    return rplt

def find_plateaus(y, tol=0.0, min_len=5, use_median=True):
    """
    Find plateaus in a 1D curve y: contiguous segments where the signal is constant
    (within tolerance), i.e. |y[i] - y[i-1]| <= tol.

    Parameters
    ----------
    y : array-like, shape (n,)
        Input signal.
    tol : float, default=0.0
        Tolerance for changes between consecutive samples. Use tol>0 for noisy signals.
    min_len : int, default=5
        Minimum plateau length (in samples) to report.
    use_median : bool, default=True
        If True, reported plateau value is median of the segment; else mean.

    Returns
    -------
    plateaus : list of dict
        Each dict has keys: start, end, length, value
        where start/end are inclusive indices.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    n = y.size
    if n == 0:
        return []

    # A "flat" step is when consecutive points differ by <= tol
    flat = np.abs(np.diff(y)) <= tol  # length n-1, True means "no change"

    # We want runs of True in `flat`. If flat[i] is True, then y[i] and y[i+1] belong together.
    # Convert flat runs into plateau segments in y indices.
    plateaus = []

    # Pad with False to detect run boundaries via diff
    f = np.concatenate([[False], flat, [False]])
    changes = np.diff(f.astype(int))
    run_starts = np.where(changes == 1)[0]      # index in flat where True-run starts
    run_ends   = np.where(changes == -1)[0] - 1 # index in flat where True-run ends

    for s_flat, e_flat in zip(run_starts, run_ends):
        # Convert flat-run [s_flat..e_flat] to y indices [start..end]
        start = s_flat
        end = e_flat + 1  # because flat length is n-1, and links y[e_flat] to y[e_flat+1]
        length = end - start + 1

        if length >= min_len:
            seg = y[start:end+1]
            value = np.median(seg) if use_median else np.mean(seg)
            plateaus.append({
                "start": int(start),
                "end": int(end),
                "length": int(length),
                "value": float(value),
            })

    return plateaus

### Wrap together all the measures over the recurrence plot - Lam and Det computed as auc for different values of l_min ###

def _run_lengths_ones(x: np.ndarray) -> np.ndarray:
    """Lengths of consecutive 1-runs in a 1D array."""
    x01 = (x != 0).astype(np.int8)  # signed matters for diff
    d = np.diff(np.r_[0, x01, 0])
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    return ends - starts

def _tail_weighted_curve_from_counts(counts: np.ndarray, lmins: np.ndarray) -> np.ndarray:
    """
    counts[r] = number of runs of length r.
    curve(lmin) = sum_{r>=lmin} r*counts[r] / sum_{r>=1} r*counts[r]
    """
    if counts.size <= 1:
        return np.zeros_like(lmins, dtype=float)

    r = np.arange(counts.size, dtype=np.float64)
    w = r * counts.astype(np.float64)

    denom = w.sum()
    if denom == 0:
        return np.zeros_like(lmins, dtype=float)

    tail = np.cumsum(w[::-1])[::-1]  # tail[l] = sum_{r>=l} w[r]
    lmins_clip = np.clip(lmins, 0, counts.size - 1)
    return tail[lmins_clip] / denom

def _mean_run_length_from_counts(counts: np.ndarray, min_len: int = 1) -> float:
    """Mean run length over runs with length >= min_len."""
    if counts.size <= 1:
        return 0.0
    min_len = max(1, int(min_len))

    r = np.arange(counts.size, dtype=np.float64)
    c = counts.astype(np.float64)

    num = np.sum(r[min_len:] * c[min_len:])
    den = np.sum(c[min_len:])
    return 0.0 if den == 0 else float(num / den)

def _entropy_from_counts(counts: np.ndarray, min_len: int = 2, log_base: float = np.e) -> float:
    """Shannon entropy over run-length distribution restricted to lengths >= min_len."""
    if counts.size <= 1:
        return 0.0
    min_len = max(1, int(min_len))

    c = counts.astype(np.float64).copy()
    c[:min_len] = 0.0
    total = c.sum()
    if total == 0:
        return 0.0

    p = c[c > 0] / total
    return float(-np.sum(p * (np.log(p) / np.log(log_base))))

def _excluded_band_count(n: int, w: int) -> int:
    """
    Number of matrix entries (i,j) with |i-j| <= w in an n x n matrix.
    """
    w = min(max(int(w), 0), n - 1)
    # sum_{k=-w..w} (n-|k|) = n(2w+1) - w(w+1)
    return int(n * (2 * w + 1) - w * (w + 1))

def rqa_all_metrics_auc_rr_theiler(
    M: np.ndarray,
    *,
    theiler: int = 0,            # exclude band |i-j| <= theiler
    # curves domain (default: 2..n-2)
    lmin_start: int = 2,
    lmin_stop: int | None = None,
    # vertical scan option (to mimic your old "range(1, n)")
    exclude_first_col: bool = False,
    # scalar params
    lmean_min: int = 3,
    tt_min: int = 3,
    ent_min: int = 2,
    log_base: float = np.e,
):
    """
    Computes in one wrapper:

    - RR with Theiler window exclusion (band removed)
    - DET(lmin) curve + AUC on lmin range
    - LAM(lmin) curve + AUC on lmin range
    - Lmax, Lmean (diagonal), TT (vertical), diagonal entropy

    Theiler window exclusion is applied consistently by treating |i-j|<=theiler
    entries as excluded (i.e., removed/zeroed) for *all* metrics.
    """

    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("M must be a 2D array.")
    n, m = M.shape
    if n != m:
        raise ValueError("M must be square.")
    if n < 5:
        raise ValueError("Need n >= 5 to have default lmin range 2..n-2.")

    w = min(max(int(theiler), 0), n - 1)

    # lmin range
    lmin_start = int(lmin_start)
    if lmin_stop is None:
        lmin_stop = n - 2
    lmin_stop = int(lmin_stop)
    lmin_start = max(1, lmin_start)
    lmin_stop = min(n - 2, lmin_stop)
    if lmin_stop < lmin_start:
        raise ValueError("Invalid lmin range: stop < start.")
    lmins = np.arange(lmin_start, lmin_stop + 1)

    # ---------- RR with Theiler exclusion ----------
    total_rec = int(np.sum(M != 0))
    excluded_rec = 0
    # subtract recurrences within the excluded band using diagonals
    for offset in range(-w, w + 1):
        excluded_rec += int(np.sum(np.diagonal(M, offset=offset) != 0))

    rr_num = total_rec - excluded_rec
    rr_den = n * n - _excluded_band_count(n, w)
    rr = 0.0 if rr_den <= 0 else rr_num / rr_den

    # ---------- Diagonal run-length histogram (excluding offsets within theiler band) ----------
    diag_counts = np.zeros(n + 1, dtype=np.int64)
    for offset in range(-(n - 1), n):
        if abs(offset) <= w:
            continue  # Theiler exclusion: remove entire offsets in the band
        rl = _run_lengths_ones(np.diagonal(M, offset=offset))
        if rl.size:
            diag_counts += np.bincount(rl, minlength=diag_counts.size)

    det_curve = _tail_weighted_curve_from_counts(diag_counts, lmins)
    det_auc = float(np.trapezoid(det_curve, x=lmins))
    l_range = float(lmins[-1] - lmins[0]) if lmins.size > 1 else 1.0
    det_auc_norm = det_auc / l_range if l_range > 0 else 0.0

    diag_nonzero = np.flatnonzero(diag_counts)
    lmax = int(diag_nonzero.max()) if diag_nonzero.size else 0
    lmean = _mean_run_length_from_counts(diag_counts, min_len=lmean_min)
    entropy = _entropy_from_counts(diag_counts, min_len=ent_min, log_base=log_base)

    # ---------- Vertical run-length histogram with Theiler exclusion (per-column masked segment) ----------
    vert_counts = np.zeros(n + 1, dtype=np.int64)
    j0 = 1 if exclude_first_col else 0

    for j in range(j0, m):
        # Excluded rows for this column are i in [j-w, j+w]
        a = max(0, j - w)
        b = min(n, j + w + 1)

        # Top segment
        if a > 0:
            rl_top = _run_lengths_ones(M[:a, j])
            if rl_top.size:
                vert_counts += np.bincount(rl_top, minlength=vert_counts.size)

        # Bottom segment
        if b < n:
            rl_bot = _run_lengths_ones(M[b:, j])
            if rl_bot.size:
                vert_counts += np.bincount(rl_bot, minlength=vert_counts.size)

    lam_curve = _tail_weighted_curve_from_counts(vert_counts, lmins)
    lam_auc = float(np.trapezoid(lam_curve, x=lmins))
    lam_auc_norm = lam_auc / l_range if l_range > 0 else 0.0

    tt = _mean_run_length_from_counts(vert_counts, min_len=tt_min)

    return {
        # meta
        "n": n,
        "theiler": w,
        "lmins": lmins,

        # RR
        "rr": float(rr),
        "rr_num": int(rr_num),
        "rr_den": int(rr_den),

        # DET/LAM curves + AUC
        "det_curve": det_curve,
        "lam_curve": lam_curve,
        "det_auc": float(det_auc),
        "lam_auc": float(lam_auc),
        "det_auc_norm": float(det_auc_norm),  # average DET across lmin-range
        "lam_auc_norm": float(lam_auc_norm),  # average LAM across lmin-range

        # diagonal scalars
        "lmax": int(lmax),
        "lmean": float(lmean),
        "entropy": float(entropy),

        # vertical scalar
        "tt": float(tt),
    }

def compute_one(ts, m, theiler, rs_perc0):
    # 1) tau
    best_tau = MI_for_delay(ts)
    # 2) embedding -> want shape (d, N) for distance_matrix_fast
    # your td_embedding seems to return (N, d); adjust if needed
    emb = td_embedding(ts, m, best_tau).T   # (N, d) ?
    # 3) distances
    D = distance_matrix_scipy(emb, m_norm=True, m=m)  # set m_norm if you actually want it

    # 4) threshold r from max distance
    max_distance = D.max()
    r = max_distance * rs_perc0

    # 5) recurrence plot (binary)
    rp = rec_plt(D, r, D.shape[0])

    # 6) RQA
    res = rqa_all_metrics_auc_rr_theiler(rp, lmin_stop=rp.shape[0]-1, theiler=theiler)
    return res

def run_subject_rqa_parallel(trials, 
                             labels, 
                             resp_points, 
                             ch_names,
                             rs_perc=(0.1,0.2,0.3,0.4), 
                             m=3, 
                             theiler=5,
                             n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count()-1))

    rs_perc0 = rs_perc[0]  # your current code uses only rs_perc[0]

    res_dicts = []
    uniq_conds = np.unique(labels)

    for cond in uniq_conds:
        bool_labels = np.asarray(labels) == cond
        trials_ = [tr[:, :resp_points[i]] for i, tr in enumerate(trials) if bool_labels[i]]
        n_trials = len(trials_)
        n_ch = len(ch_names)

        res_dict = {
            "rr": np.zeros((n_trials, n_ch)),
            "det_auc_norm": np.zeros((n_trials, n_ch)),
            "lam_auc_norm": np.zeros((n_trials, n_ch)),
            "lmax": np.zeros((n_trials, n_ch)),
            "lmean": np.zeros((n_trials, n_ch)),
            "entropy": np.zeros((n_trials, n_ch)),
            "tt": np.zeros((n_trials, n_ch)),
        }

        # Build tasks (trial_i, ch_idx, ts)
        tasks = []
        for trial_i, trial in enumerate(trials_):
            for ch_idx in range(n_ch):
                tasks.append((trial_i, ch_idx, trial[ch_idx]))

        # Parallel compute
        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(compute_one)(ts, m, theiler, rs_perc0)
            for (_, _, ts) in tqdm(tasks, desc=f"cond={cond} tasks")
        )

        # Fill outputs
        for (trial_i, ch_idx, _), rqa_res in zip(tasks, results):
            for key in res_dict.keys():
                if key in rqa_res:
                    res_dict[key][trial_i, ch_idx] = rqa_res[key]

        res_dicts.append(res_dict)

    return res_dicts

def prepare_pair_distances(dist_matrix: np.ndarray, theiler: int = 0) -> np.ndarray:
    """
    Extract valid pair distances (i>j), optionally excluding |i-j| <= theiler.
    Returns a 1D array of distances.
    """
    D = np.asarray(dist_matrix)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_matrix must be square (N,N).")

    N = D.shape[0]

    # indices for lower triangle (i>j)
    ii, jj = np.tril_indices(N, k=-1)

    if theiler > 0:
        keep = (ii - jj) > theiler
        ii, jj = ii[keep], jj[keep]

    d = D[ii, jj]
    # optional: drop zeros if they are self-distances or duplicates
    # (you already exclude diagonal; zeros may still exist if identical points)
    d = d[d > 0]

    return d


def corr_sum_many_rs(dist_matrix: np.ndarray, rs: np.ndarray, theiler: int = 0) -> np.ndarray:
    """
    Compute correlation sum C(r) for many r values efficiently.

    C(r) = (#pairs with distance < r) / (#valid pairs)
    """
    rs = np.asarray(rs, dtype=float)
    if np.any(rs <= 0):
        raise ValueError("All r must be > 0 to take logs later.")

    d = prepare_pair_distances(dist_matrix, theiler=theiler)
    if d.size == 0:
        raise ValueError("No valid pairs after Theiler exclusion / filtering.")

    d_sorted = np.sort(d)
    counts = np.searchsorted(d_sorted, rs, side="left")
    C = counts / d_sorted.size
    return C

from scipy.ndimage import gaussian_filter1d

def rolling_linreg_slope(x: np.ndarray, y: np.ndarray, win: int) -> np.ndarray:
    """
    Rolling linear regression slope of y ~ a*x + b over a centered window of length win.
    Returns an array of length len(x)-win+1 aligned to window centers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if win < 2 or win > x.size:
        raise ValueError("win must be in [2, len(x)].")

    w = np.ones(win)

    Sx  = np.convolve(x, w, mode="valid")
    Sy  = np.convolve(y, w, mode="valid")
    Sxx = np.convolve(x*x, w, mode="valid")
    Sxy = np.convolve(x*y, w, mode="valid")

    denom = win * Sxx - Sx*Sx
    # avoid division by zero
    denom = np.where(denom == 0, np.nan, denom)

    slope = (win * Sxy - Sx*Sy) / denom
    return slope


def corr_exponent_from_C(rs: np.ndarray,
                         Cs: np.ndarray,
                         win: int = 50,
                         smooth_sigma: float | None = None):
    """
    Compute local correlation exponent via rolling regression on log-log curve.
    Returns: log_r_center, exponent
    """
    rs = np.asarray(rs, float)
    Cs = np.asarray(Cs, float)

    # avoid log(0)
    eps = np.finfo(float).tiny
    log_r = np.log(rs)
    log_C = np.log(np.maximum(Cs, eps))

    expn = rolling_linreg_slope(log_r, log_C, win=win)

    # x-values aligned to window centers
    centers = np.arange(win//2, log_r.size - (win - 1)//2)
    log_r_center = log_r[centers]

    if smooth_sigma is not None and smooth_sigma > 0:
        expn = gaussian_filter1d(expn, sigma=smooth_sigma)

    return log_r_center, expn

### Compute only embedded time-series and recurrence plot ###
def compute_one_ts_emb_RP(ts, 
                          m,
                          tau, 
                          rs_perc0):

    # 2) embedding -> want shape (d, N) for distance_matrix_fast
    # your td_embedding seems to return (N, d); adjust if needed
    emb = td_embedding(ts, m, tau).T   # (N, d) ?
    d, N = emb.shape

    # 3) distances
    D = distance_matrix_scipy(emb, m_norm=True, m=m)  # set m_norm if you actually want it

    # 4) threshold r from max distance
    max_distance = D.max()
    r = max_distance * rs_perc0

    # 5) recurrence plot (binary)
    rp = rec_plt(D, r, D.shape[0])

    return {'emb':emb, 'RP':rp.astype(np.uint8)}


## RQA and D2 an Lyapunov merged together ##
def compute_one_ts_RQA_D2_Ly(ts, 
                          m,
                          theiler, 
                          rs_perc0,    
                          best_tau: int = 0,
                          tol: float = 0.01,
                          min_len: int = 20,
                          n_rs: int = 1000,
                          prc_min: float = 1.0,
                          prc_max: float = 99.0,
                          win: int = 50,
                          min_embedded_points: int = 20,
                          smooth_sigma: float | None = 1.0):
    # 1) tau
    if best_tau==0:
        best_tau = MI_for_delay(ts)
    # 2) embedding -> want shape (d, N) for distance_matrix_fast
    # your td_embedding seems to return (N, d); adjust if needed
    emb = td_embedding(ts, m, best_tau).T   # (N, d) ?
    d, N = emb.shape

    # ensure rolling window is valid
    if win >= N:
        # shrink window to something valid
        win_eff = max(5, min(N - 1, win))
    else:
        win_eff = win

    # 3) distances
    D = distance_matrix_scipy(emb, m_norm=True, m=m)  # set m_norm if you actually want it

    # 4) threshold r from max distance
    max_distance = D.max()
    r = max_distance * rs_perc0

    # 5) recurrence plot (binary)
    rp = rec_plt(D, r, D.shape[0])

    # 6) RQA
    res = rqa_all_metrics_auc_rr_theiler(rp, lmin_stop=rp.shape[0]-1, theiler=theiler)
    # 7) D2
    
    # 7.1) choose radii via percentiles of valid pair distances (with same Theiler)
    try:
        d_all = prepare_pair_distances(D, theiler=theiler)
    except Exception:
        return res

    if d_all.size < 10:
        return res

    min_r = np.percentile(d_all, prc_min)
    max_r = np.percentile(d_all, prc_max)

    # guard against degenerate ranges
    if not np.isfinite(min_r) or not np.isfinite(max_r) or min_r <= 0 or max_r <= min_r:
        return res

    rs = np.logspace(np.log10(min_r), np.log10(max_r), int(n_rs))

    # 7.2) C(r)
    try:
        Cs = corr_sum_many_rs(D, rs, theiler=theiler)
    except Exception:
        return res

    # 7.3) local exponent
    try:
        log_r_center, expn = corr_exponent_from_C(rs, Cs, win=win_eff, smooth_sigma=smooth_sigma)
    except Exception:
        return res

    if expn.size == 0 or not np.any(np.isfinite(expn)):
        return res

    # 7.4) plateau detection (your function)
    try:
        plateaus = find_plateaus(expn, tol=tol, min_len=min_len)
    except Exception:
        return res

    if not plateaus:
        return res

    # Append the D2 value to the RQA results
    p0 = plateaus[0]
    # print(p0["value"])
    res["d2"] = float(p0["value"])

    # 8) Lyapunov
    f, ps = periodogram(emb)
    dt = int(np.sum(ps)/np.sum(f*ps))
    ly, e_ly, _, _, _  = lyap(dist_matrix=D, dt=dt, w=theiler, T=emb.shape[1], sfreq=1000, verbose=False)
    # print(ly)
    res["ly"] = ly

    return res

def run_subject_RQA_D2_Ly_parallel(
    trials,
    labels,
    resp_points,
    ch_names,
    *,
    rs_perc=(0.1,0.2,0.3,0.4),
    m: int = 3,
    theiler: int = 5,
    win: int = 50,
    smooth_sigma: float | None = 1.0,
    tol: float = 0.002,
    min_len: int = 20,
    n_rs: int = 1000,
    prc_min: float = 1.0,
    prc_max: float = 99.0,
    min_embedded_points: int = 20,
    n_jobs: int | None = None,
):
    """
    Returns: list of dicts, one per condition, each containing arrays:
      - d2 (n_trials_cond, n_channels)
      - tau (n_trials_cond, n_channels)
      - n_emb (n_trials_cond, n_channels)
      - plateau_found (bool array)
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) // 2)

    rs_perc0 = rs_perc[0]  # your current code uses only rs_perc[0]

    res_dicts = [] # Two dictionary inside this list, one per condition
    uniq_conds = np.unique(labels)

    for cond in uniq_conds:
        bool_labels = np.asarray(labels) == cond
        trials_ = [tr[:, :resp_points[i]] for i, tr in enumerate(trials) if bool_labels[i]]

        n_trials = len(trials_)
        n_ch = len(ch_names)

        res = {
            "rr": np.zeros((n_trials, n_ch)),
            "det_auc_norm": np.zeros((n_trials, n_ch)),
            "lam_auc_norm": np.zeros((n_trials, n_ch)),
            "lmax": np.zeros((n_trials, n_ch)),
            "lmean": np.zeros((n_trials, n_ch)),
            "entropy": np.zeros((n_trials, n_ch)),
            "tt": np.zeros((n_trials, n_ch)),
            "d2": np.zeros((n_trials, n_ch)),
            "ly": np.zeros((n_trials, n_ch))
        }

        tasks = []
        for trial_i, trial in enumerate(trials_):
            for ch_idx in range(n_ch):
                tasks.append((trial_i, ch_idx, trial[ch_idx]))

        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(compute_one_ts_RQA_D2_Ly)(
                ts,
                m=m,
                theiler=theiler,
                win=win,
                rs_perc0=rs_perc0, 
                smooth_sigma=smooth_sigma,
                tol=tol,
                min_len=min_len,
                n_rs=n_rs,
                prc_min=prc_min,
                prc_max=prc_max,
                min_embedded_points=min_embedded_points,
            )
            for (_, _, ts) in tqdm(tasks, desc=f"D2 cond={cond} tasks")
        )

        # Fill outputs
        for (trial_i, ch_idx, _), rqa_res in zip(tasks, results):
            for key in res.keys():
                if key in rqa_res:
                    res[key][trial_i, ch_idx] = rqa_res[key]

        res_dicts.append(res)

    return res_dicts

### Parallel computation for embedded time-series and recurrence plot only ###
def hardcore_parallel_emb_RP(
    subj_list,
    task, # breaking or reverse
    ch_names,
    *,
    rs_perc=(0.1,0.2,0.3,0.4),
    m: int = 3,
    n_tp: int = n_timepoints,
    tau: int = best_mean_tau,
    n_jobs: int | None = None):

    """""
    Parallel also the loop over the subjects
    """""
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) // 2)

    rs_perc0 = rs_perc[0]  # your current code uses only rs_perc[0]
    n_ch = len(ch_names)

    res_by_subj = []
    tasks = []
    for subj_i, subj in enumerate(subj_list):
        if task=='breaking':
            file_path = data_dir / task / f"subj_{subj}_breakingtrials_dict_TRfilt_1500.pkl"
        else:
            file_path = data_dir / task / f"subj_{subj}_reversetrials_dict_TRfilt_1500.pkl"

        with open(file_path, 'rb') as handle:
            trials_dict = pickle.load(handle)
        trials = trials_dict['trial']
        resp_points = trials_dict['resp_point']
        labels = trials_dict['label']
        labels_unq = np.unique(labels)
        ## Select only the face trials with time after the response ##
        bool_idx = np.array(labels)==labels_unq[1]
        face_trials = [trial for i, trial in enumerate(trials) if bool_idx[i]]
        face_RTs = np.array(resp_points)[bool_idx] 
        count = 0
        valid_idx = [] # Store the index to pick the matched control trials
        for trial_i, trial in enumerate(face_trials):
            resp_point = face_RTs[trial_i]
            # trial_ = trial[:,resp_point-500:resp_point+500] # Around the resp
            trial_ = trial[:,:1000] # Start
            if trial_.shape[1]!=0:
                valid_idx.append(trial_i)
                for ch_idx in range(n_ch):
                    tasks.append((subj, count, labels_unq[1], ch_idx, trial_[ch_idx]))
                count += 1
        ## Pick the control trials matched with the valid face trials ##
        bool_idx = np.array(labels)==labels_unq[0]
        contr_trials = [trial for i, trial in enumerate(trials) if bool_idx[i]]
        contr_RTs = np.array(resp_points)[bool_idx]
        count = 0
        for idx in valid_idx:
            trial = contr_trials[idx]
            resp_point = contr_RTs[idx]
            # trial_ = trial[:,resp_point-500:resp_point+500] # Around the resp
            trial_ = trial[:,:1000] # start
            for ch_idx in range(n_ch):
                tasks.append((subj, count, labels_unq[0], ch_idx, trial_[ch_idx]))
            count += 1
        # Initialize the dictionaries to store the results for the current subject
        res_face = {
        "emb": np.zeros((len(valid_idx), n_ch, n_tp)),
        "RP": np.zeros((len(valid_idx), n_ch, n_tp, n_tp))
        }
        res_contr = {
        "emb": np.zeros((len(valid_idx), n_ch, n_tp)),
        "RP": np.zeros((len(valid_idx), n_ch, n_tp,n_tp))
        }
        res_by_subj.append([res_face, res_contr])
    print('Starting the '+task+' computation...')
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes",verbose=5)(
                        delayed(compute_one_ts_emb_RP)(
                                ts,
                                m=m,
                                tau=tau,
                                rs_perc0=rs_perc0,
                                )
                            for (_, _, _, _, ts) in tqdm(tasks, desc=f"Task={task}")
                        )
    # Fill outputs
    for (subj_, trial_i, task_, ch_idx, _), res in zip(tasks, results): # Loop over single result
        subj_i = np.where(np.array(subj_list)==subj_)[0][0]
        cond_i = np.where(np.array([labels_unq[1],labels_unq[0]])==task_)[0][0]
        for key in res_by_subj[subj_i][cond_i].keys():
            if key in res:
                res_by_subj[subj_i][cond_i][key][trial_i, ch_idx] = res[key]


    return res_by_subj

def hardcore_parallel(
    subj_list,
    task, # breaking or reverse
    ch_names,
    position: str = 'AroundResp',
    *,
    rs_perc=(0.1,0.2,0.3,0.4),
    m: int = 3,
    best_tau: int = 0,
    theiler: int = 5,
    win: int = 50,
    smooth_sigma: float | None = 1.0,
    tol: float = 0.002,
    min_len: int = 20,
    n_rs: int = 1000,
    prc_min: float = 1.0,
    prc_max: float = 99.0,
    min_embedded_points: int = 20,
    n_jobs: int | None = None):

    """""
    Parallel also the loop over the subjects
    """""
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) // 2)

    rs_perc0 = rs_perc[0]  # your current code uses only rs_perc[0]
    res_dicts = []
    n_ch = len(ch_names)

    res_by_subj = []
    tasks = []
    for subj_i, subj in enumerate(subj_list):
        if task=='breaking':
            file_path = data_dir / task / f"subj_{subj}_breakingtrials_dict_TRfilt_1500.pkl"
        else:
            file_path = data_dir / task / f"subj_{subj}_reversetrials_dict_TRfilt_1500.pkl"

        with open(file_path, 'rb') as handle:
            trials_dict = pickle.load(handle)
        trials = trials_dict['trial']
        resp_points = trials_dict['resp_point']
        labels = trials_dict['label']
        labels_unq = np.unique(labels)
        ## Select only the face trials with time after the response ##
        bool_idx = np.array(labels)==labels_unq[1]
        face_trials = [trial for i, trial in enumerate(trials) if bool_idx[i]]
        face_RTs = np.array(resp_points)[bool_idx] 
        count = 0
        valid_idx = [] # Store the index to pick the matched control trials
        for trial_i, trial in enumerate(face_trials):
            resp_point = face_RTs[trial_i]
            if position=='AroundResp':
                trial_ = trial[:,resp_point-500:resp_point+500] # Around the resp
            elif position=='Start':
                trial_ = trial[:,:1000] # Start
            if trial_.shape[1]==1000:
                valid_idx.append(trial_i)
                for ch_idx in range(n_ch):
                    tasks.append((subj, count, labels_unq[1], ch_idx, trial_[ch_idx]))
                count += 1
        ## Pick the control trials matched with the valid face trials ##
        bool_idx = np.array(labels)==labels_unq[0]
        contr_trials = [trial for i, trial in enumerate(trials) if bool_idx[i]]
        contr_RTs = np.array(resp_points)[bool_idx]
        count = 0
        for idx in valid_idx:
            trial = contr_trials[idx]
            resp_point = contr_RTs[idx]
            if position=='AroundResp':
                trial_ = trial[:,resp_point-500:resp_point+500] # Around the resp
            elif position=='Start':
                trial_ = trial[:,:1000] # start
            for ch_idx in range(n_ch):
                tasks.append((subj, count, labels_unq[0], ch_idx, trial_[ch_idx]))
            count += 1
        # Initialize the dictionaries to store the results for the current subject
        res_face = {
        "rr": np.zeros((len(valid_idx), n_ch)),
        "det_auc_norm": np.zeros((len(valid_idx), n_ch)),
        "lam_auc_norm": np.zeros((len(valid_idx), n_ch)),
        "lmax": np.zeros((len(valid_idx), n_ch)),
        "lmean": np.zeros((len(valid_idx), n_ch)),
        "entropy": np.zeros((len(valid_idx), n_ch)),
        "tt": np.zeros((len(valid_idx), n_ch)),
        "d2": np.zeros((len(valid_idx), n_ch)),
        "ly": np.zeros((len(valid_idx), n_ch))
        }
        res_contr = {
        "rr": np.zeros((len(valid_idx), n_ch)),
        "det_auc_norm": np.zeros((len(valid_idx), n_ch)),
        "lam_auc_norm": np.zeros((len(valid_idx), n_ch)),
        "lmax": np.zeros((len(valid_idx), n_ch)),
        "lmean": np.zeros((len(valid_idx), n_ch)),
        "entropy": np.zeros((len(valid_idx), n_ch)),
        "tt": np.zeros((len(valid_idx), n_ch)),
        "d2": np.zeros((len(valid_idx), n_ch)),
        "ly": np.zeros((len(valid_idx), n_ch))
        }
        res_by_subj.append([res_face, res_contr])
    print('Starting the '+task+' computation...')
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes",verbose=5)(
                        delayed(compute_one_ts_RQA_D2_Ly)(
                                ts,
                                m=m,
                                best_tau=best_tau,
                                theiler=theiler,
                                win=win,
                                rs_perc0=rs_perc0, 
                                smooth_sigma=smooth_sigma,
                                tol=tol,
                                min_len=min_len,
                                n_rs=n_rs,
                                prc_min=prc_min,
                                prc_max=prc_max,
                                min_embedded_points=min_embedded_points,
                                )
                            for (_, _, _, _, ts) in tqdm(tasks, desc=f"Task={task}")
                        )
    # Fill outputs
    for (subj_, trial_i, task_, ch_idx, _), rqa_res in zip(tasks, results):
        subj_i = np.where(np.array(subj_list)==subj_)[0][0]
        cond_i = np.where(np.array([labels_unq[1],labels_unq[0]])==task_)[0][0]
        for key in res_by_subj[subj_i][cond_i].keys():
            if key in rqa_res:
                res_by_subj[subj_i][cond_i][key][trial_i, ch_idx] = rqa_res[key]


    return res_by_subj


### Load the montage ###
# montage = mne.channels.read_dig_fif('new_montage.fif')
# montage = mne.channels.read_dig_fif('../../data/new_montage.fif')
montage = mne.channels.read_dig_fif(data_dir / 'new_montage.fif')

info = mne.create_info(ch_names=montage.ch_names, sfreq=1000, ch_types='eeg')
ch_names = montage.ch_names
info.set_montage(montage);
print('Montage loaded...')

### Run RQA analysis for all the subjects ###

subj_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14',
             '15','16','17','18','19','20','21','22','23','25','26','27','28',
             '29','30','31','32','33','34','35']
# subj_list = ['01','02','03']

main_path = '../../data/prep_task_Marina/'
# main_path = 'E:/PhD/CFS_eeg/data/new_exp/prep_task_Marina/'

break_or_revs =  ['breaking','reverse']
break_or_revs_ = ['','rev']


if not HARDCORE_PARALLEL and not HARDCORE_PARALLEL_EMB_RP:
    breaking_res_D2_rqa = [] 
    reverse_res_D2_rqa = []

    for subj in subj_list:
        for break_or_rev, break_or_rev_ in zip(break_or_revs, break_or_revs_) :
            if break_or_rev=='breaking':
                file_path = main_path+break_or_rev+'/bCFSsubj_'+subj+'trials_dict.pkl'
            else:
                file_path = main_path+break_or_rev+'/revsubj_'+subj+'_rev_trials_dict.pkl'

            with open(file_path, 'rb') as handle:
                trials_dict = pickle.load(handle)
            trials = trials_dict['trial']
            resp_points = trials_dict['resp_point']
            labels = trials_dict['label']
            baselines = trials_dict['baseline']

            res_by_cond = run_subject_RQA_D2_Ly_parallel(trials, 
                                                            labels, 
                                                            resp_points, 
                                                            ch_names,
                                                            m=3,
                                                            theiler=5,
                                                            win=50,
                                                            smooth_sigma=1.0,
                                                            tol=0.002,
                                                            min_len=20,
                                                            n_jobs=n_jobs
                                                            )
            
            if break_or_rev=='breaking':
                breaking_res_D2_rqa.append(res_by_cond)
            else:
                reverse_res_D2_rqa.append(res_by_cond)

    with open('breaking_res_D2_rqa.pkl', 'wb') as handle:
        pickle.dump(breaking_res_D2_rqa, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('reverse_res_D2_rqa.pkl', 'wb') as handle:
        pickle.dump(reverse_res_D2_rqa, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif HARDCORE_PARALLEL and not HARDCORE_PARALLEL_EMB_RP:
    # main_path = 'E:/PhD/CFS_eeg/data/new_exp/prep_task_Marina_filtTR_1500/'
    position='AroundResp'
    for task in break_or_revs:
        results_ = hardcore_parallel(subj_list, 
                                    task, 
                                    ch_names,
                                    position=position,
                                    m=3,
                                    best_tau=best_mean_tau,
                                    theiler=5,
                                    win=50,
                                    smooth_sigma=1.0,
                                    tol=0.002,
                                    min_len=20,
                                    n_jobs=n_jobs
                                    )
        
        # with open(f"{task}_allsubj_AroundResp_D2_rqa_ly.pkl", 'wb') as handle:
        #     pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(f"{task}_allsubj_Start_D2_rqa_ly.pkl", 'wb') as handle:
        with open(f"{task}_allsubj_{position}_fixedtau_D2_rqa_ly.pkl", 'wb') as handle:
            pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif HARDCORE_PARALLEL_EMB_RP and not HARDCORE_PARALLEL:
    for task in break_or_revs:
        results_ = hardcore_parallel_emb_RP(subj_list, 
                                    task, 
                                    ch_names,
                                    m=3,
                                    n_jobs=n_jobs
                                    )
        
        # with open(f"{task}_allsubj_AroundResp_D2_rqa_ly.pkl", 'wb') as handle:
        #     pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{task}_allsubj_Start_emb_RP.pkl", 'wb') as handle:
            pickle.dump(results_, handle, protocol=pickle.HIGHEST_PROTOCOL)