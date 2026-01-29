import mne

from core import extractTS

# Get evoked signals 
def evokeds(MNE: mne.Evoked, sMNE: mne.Evoked,
            ch_list: list|tuple, window = [None, None]):

    TS, E_TS = extractTS(MNE = MNE, sMNE = sMNE, ch_list = ch_list, window = window, clst_method = 'mean')

    # Initzialize result array
    EP = []
    E_EP = []

    # Loop around pois time series
    for ts, e_ts in zip(TS, E_TS):
        for t, e_t in zip(ts, e_ts):

            EP.append(t)
            E_EP.append(e_t)

    return EP, E_EP

# Iterable function generator
def it_evokeds(info: dict):

    global iterable

    # Build Evoked Plotting iterable function
    def iterable(MNE_l: list):

        EP, E_EP = evokeds(MNE = MNE_l[0], sMNE = MNE_l[1],
                           ch_list = info['ch_list'],
                           window = info['window'])

        return EP, E_EP

    return iterable
