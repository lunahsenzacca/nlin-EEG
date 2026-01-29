import warnings
import numpy as np

from scipy.stats import linregress

from core import gauss_kernel, to_log, obs_path, loadresults, flat_results

# Correlation Exponent elementar function
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
            results = linregress(x = np.asarray(log_r)[i:i + n_points], y = log_csum[i:i + n_points], nan_policy = 'omit')

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

# Prepare corrsum.py results for correlation_exponent sub-wise function
def correxp_getcorrsum(info: dict, load_calc_lb: str):

    path = obs_path(exp_name = info['exp_name'], obs_name = 'corrsum', avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = load_calc_lb)

    # Load correlation sum results
    CS, _, variables = loadresults(obs_path = path, obs_name = 'corrsum', X_transform = None)

    flat_CS, points = flat_results(CS)

    # Initzialize trial-wise iterable
    log_CS_iters = []
    for arr in flat_CS:

        log_CS = to_log(arr, verbose = False)

        # Build trial-wise iterable
        log_CS_iters.append([log_CS[0],log_CS[1]])

    return log_CS_iters, points, variables

# Sub-wise function for Correlation Exponent computation
def correlation_exponent(log_CS_iters: list, n_points: int, log_r: list,
                         gauss_filter = False, scale = None, cutoff = None):

    # Reduced rvals lenght for mobile average
    rlen = len(log_r) - n_points + 1

    # Initzialize results arrays
    CE = []
    E_CE = []

    abc = log_CS_iters[0]
    abc_ = log_CS_iters[1]
    for ab, ab_ in zip(abc, abc_):
        for a, a_ in zip(ab, ab_):

            ce, e_ce = corr_exp(log_csum = a, log_r = log_r, n_points = n_points, gauss_filter = gauss_filter, scale = scale, cutoff = cutoff)

            CE.append(ce)
            E_CE.append(e_ce)

    CE = np.asarray(CE)
    E_CE = np.asarray(E_CE)

    rshp = list(log_CS_iters[0].shape)

    rshp[-1] = rlen

    CE = CE.reshape(rshp)
    E_CE = E_CE.reshape(rshp)

    return CE, E_CE

# Iterable function generator
def it_correlation_exponent(variables: dict):

    global iterable

    def iterable(log_CS_iters: np.ndarray):

        CE, E_CE = correlation_exponent(log_CS_iters = log_CS_iters,
                                        n_points = variables['n_points'],
                                        gauss_filter = variables['gauss_filter'],
                                        scale = variables['scale'],
                                        cutoff = variables['cutoff'],
                                        log_r = variables['log_r'])

        return CE, E_CE

    return iterable
