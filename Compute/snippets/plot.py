def plot_head(M: np.ndarray, exp_lb: str, axes = Axes, cmap: str = 'coolwarm', vlim: tuple = (None,None)):

    ex_file = os.listdir(f'../Cargo/toMNE/avg/{exp_lb}')[0]
    _ = mne.read_evokeds(f'../Cargo/toMNE/avg/{exp_lb}/{ex_file}', verbose = False)[0]

    info = _.info

    del _

    im, cou = mne.viz.plot_topomap(M, info, axes = axes, show = False, cmap = cmap, sensors = 'k*', vlim = vlim)

    return im
