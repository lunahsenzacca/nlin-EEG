import warnings
import numpy as np

from scipy.stats import linregress

from core import gauss_kernel, obs_path

# Correlation Exponent elementar function
def correlation_exponent(CS: list | np.ndarray, log_r: list, n_points: int, gauss_filter: bool, scale = None, cutoff: int | None = None, log_CS: bool = False):

    if log_CS is False:
        for i, cs in CS:
            CS[i] = np.log(cs)

    rlen = len(log_r) - n_points + 1

    ce =  []
    e_ce = []
    n_log_r = []
    for i in range(0,rlen):

        m = np.array([(CS[i+j+1] - CS[i+j])/(log_r[i+j+1] - log_r[i+j]) for j in range(0,n_points-1)])

        # Get value for error of slope
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = linregress(x = np.asarray(log_r)[i:i + n_points], y = CS[i:i + n_points], nan_policy = 'omit')

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
def get_corrsum(info: dict, load_calc_lb: str):

    path = obs_path(exp_name = info['exp_name'], obs_name = 'recurrence', avg_trials = info['avg_trials'], clst_lb = info['clst_lb'], calc_lb = load_calc_lb)

    # Load Correlation Sum file
    CS = np.load(path + 'corrsum.npz')

    return CS

# Iterable function generator
def it_correlation_exponent(variables: dict):

    global iterable

    def iterable(CS: np.ndarray):

        CE, E_CE = correlation_exponent(CS = CS,
                                        n_points = variables['n_points'],
                                        gauss_filter = variables['gauss_filter'],
                                        scale = variables['scale'],
                                        cutoff = variables['cutoff'],
                                        log_r = variables['log_r'])

        return CE, E_CE

    return iterable
