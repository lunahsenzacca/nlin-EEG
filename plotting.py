# Usual suspects
import os
import json
import warnings

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as mcolors

# Utility functions for directories and data
from core import pics_path, obs_path, obs_data

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### Extra DICTIONARIES (Eventually put in init.py)###

avg = {'pois': 2,
       'sub_pois': (0,2),
       'sub': 0,}

clust_dict = {'G': ['Global'],
              'Gavg': ['Global Average'],
              'mG': ['Global (m)'],
              'mGavg': ['Global Average (m)'],
              'Gsystem': ['Global System'],
              'mGsystem': ['Global System (m)'],
              'PO': ['Parieto-Occipital'],
              'F': ['Frontal'],
              'CFPO': ['Fp1','Fp2','Fpz','Frontal','O2','PO4','PO8','Parieto-Occipital','Fronto-Parieto-Occipital System'],
              'mCFPO': [ i + ' (m)' for i in ['Fp1','Fp2','Fpz','Frontal','O2','PO4','PO8','Parieto-Occipital', 'Fronto-Parieto-Occipital System']],
              'znoisefree': ['Lorenz'],
              'm_znoisefree': ['Lorenz (m)'],
              'm_znoisefree_dense': ['Lorenz (m)'],
              'gnoise': ['Gaussian Noise'],
              'm_gnoise': ['Gaussian Noise (m)'],
              'mCFPOVANdense':[ i + ' (m, VAN)' for i in ['Fp1','Fp2','Fpz','Frontal','O2','PO4','PO8','Parieto-Occipital','Fronto-Parieto-Occipital System']],
              'mCFPOdense':[ i + ' (m)' for i in ['Fp1','Fp2','Fpz','Frontal','O2','PO4','PO8','Parieto-Occipital','Fronto-Parieto-Occipital System']],
              'test': ['test']}

obs_dict = {'corrsum': '$C_{m}(r)$ ',
            'correxp': '$\\nu_{m}(r)$ ',
            'idim': '$D_{2}(m)$ ',
            'llyap': '$\\lambda(m)$ '}

cond_dict = {'S__': 'Conscious',
             'S_1': 'Unconscious',
             'lorenz': 'Lorenz',
             'noise': 'Noise'}


### PLOTTING WRAPPERS ###

def show_figure(fig):

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    return

def plot_1dfunction(OBS: np.ndarray, E_OBS: np.ndarray, X: list, multi_idxs: list, label: list, label_idxs: list, alpha_m: float, grid: list, figsize: list):

    # Axis 0 = Multiplot
    # Axis 1 = Legend
    # Axis 2 = X value

    obs = OBS[:,:,:]
    e_obs = E_OBS[:,:,:]

    cmap = cm.viridis

    norm = mcolors.Normalize(vmin = 0, vmax = len(label_idxs) - 1)

    fig, axs = plt.subplots(grid[0], grid[1], figsize = figsize, sharex = True, sharey = True)

    if len(multi_idxs) == 1:
        ax_iter = [axs]
    else:
        ax_iter = axs.flat

    for j, ax in zip(multi_idxs, ax_iter):

        for i, c in enumerate(label_idxs):

            color = cmap(norm(i))

            if j == 0:
                ax.plot(X, obs[j,c,:], color = color, alpha = 1*alpha_m, label = label[c])
            else:
                ax.plot(X, obs[j,c,:], color = color, alpha = 1*alpha_m)
            
            ax.fill_between(X, obs[j,c,:]-e_obs[j,c,:], obs[j,c,:]+e_obs[j,c,:], color = color, alpha = 0.4*alpha_m)

    plt.close()

    return fig, axs


# Main wrapper for function over pois
def plot_observable(info: dict, instructions: dict, show = True, save = False, verbose = True):

    # Legend

    # Info
    #
    #'exp_name'
    #'avg_trials'
    #'obs_name'
    #'clust_lb'
    #'calc_lb'

    # Instructions
    #
    # 'avg'
    # 'confront'
    # 'grid'
    # 'figsize'
    # 'titlesz'
    # 'sv_name'

    # Scalar value 4/5-dimensional array

    # Axis 0 = Subjects
    # Axis 1 = Conditions
    # Axis 2 = Electrodes/Clusters
    # Axis 3 = X value
    # Axis 4 = Y value

    # Value of the array is y(x) or e_y(x)

    # Check for confliction instructions
    check_conflict = [instructions['figure'], instructions['multiplot'], instructions['legend'], instructions['axis']]
    if len(set(check_conflict)) != len(check_conflict):
        print('Conflicting instructions,\'figure\', \'multiplot\', \'legend\' and \'axis\', should all be different')
        return

    # Get observable file path
    # Get relevant paths
    path = obs_path(
                       exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       )

    sv_path = pics_path(
                        exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       ) + instructions['avg'] + '/'

    # Load results for specific observable
    OBS, E_OBS, X, variables = obs_data(obs_path = path, obs_name = info['obs_name'], compound_error = info['compound_error'])

    if verbose == True:
        print(variables)

    clst = variables['clustered']

    conditions = variables['conditions']

    labels = [variables['subjects'],[cond_dict[i] for i in variables['conditions']],variables['pois'],variables['embeddings'],[instructions['e_title']]]

    if clst == True:
        labels[2] = clust_dict[info['clust_lb']]

    # Add extra dimension for consistency
    if len(OBS.shape) == 4:
        
        OBS = np.expand_dims(OBS, axis = 4)
        E_OBS = np.expand_dims(E_OBS, axis = 4)
        
    # Apply instructions for averaging
    if instructions['avg'] != 'none':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            norm = OBS.shape[avg[instructions['avg']]]

            if type(norm) != int:
                norm = norm[0]*norm[1]
            
            OBS = np.nanmean(OBS, axis = avg[instructions['avg']])
            E_OBS = np.sqrt(np.nansum(E_OBS**2, axis = avg[instructions['avg']]))/norm
            
            OBS = np.expand_dims(OBS, axis = avg[instructions['avg']])
            E_OBS = np.expand_dims(E_OBS, axis = avg[instructions['avg']])

    if instructions['avg'] == 'sub_pois' or 'pois':

        labels[2] = clust_dict[info['clust_lb']]

    # Initzialize list for array rearranging
    rearrange = [0,0,0,0,0]

    # Check figures
    if instructions['figure'] == 'pois':

        rearrange[1] = 2

        title_l = labels[2]

    elif instructions['figure'] == 'conditions':

        rearrange[1] = 1

        title_l = labels[1]

    #elif instructions['figures'] == 'conditions_pois': 

    # Selet multiplot axis
    if instructions['multiplot'] == 'subjects':

        rearrange[2] = 0

        plot_l = labels[0]

    if instructions['legend'] == 'conditions':

        rearrange[3] = 1

        legend_l = labels[1]

    elif instructions['legend'] == 'embeddings':

        rearrange[3] = 3

        legend_l = labels[3]

    if instructions['axis'] == 'embeddings':

        rearrange[4] = 3

        X = X[0]

        instructions['xlabel'] = '$m$'

    elif instructions['axis'] == 'log_r':

        rearrange[4] = 4

        X = X[1]

        instructions['xlabel'] = '$\\log(r)$'

    for i in range(0,len(rearrange)):
        if i not in rearrange:
            rearrange[0] = i

    if instructions['reduce_legend'] != None:

        label_idxs = instructions['reduce_legend']

    else:

        label_idxs = [i for i in range(0,len(legend_l))]

    if instructions['reduce_multi'] != None:

        multi_idxs = instructions['reduce_multi']

    else:

        multi_idxs = [i for i in range(0,len(plot_l))]

    # Rearrange arrays
    obs = np.permute_dims(OBS, rearrange)
    e_obs = np.permute_dims(E_OBS, rearrange)
    
    # Reduce extra axis
    if instructions['reduce_method'] == 'product':
        
        obs = [i for ob in obs for i in ob]
        e_obs = [i for e_ob in e_obs for i in e_ob]

    title_l = [str(i) + '; ' + str(j) for i in labels[rearrange[0]] for j in title_l]

    print(np.asarray(obs).shape)

    # Create iteration around figures
    OBS = [i for i in obs]
    E_OBS = [i for i in e_obs]

    # Initzialize lists for figures and axes
    figs = []
    axes = []
    for i in range(0, len(obs)):
    
        # Initialize figures
        fig, axis = plot_1dfunction(OBS = OBS[i], E_OBS = E_OBS[i], X = X, multi_idxs = multi_idxs, label = legend_l, label_idxs = label_idxs, alpha_m = instructions['alpha_m'], grid = instructions['grid'], figsize = instructions['figsize'])

        figs.append(fig)
        axes.append(axis)

    # Cycle around axis and figures to add informations

    grid = instructions['grid']
    # Check if we have multiple axes and initialize proper iterable
    for axs in axes:
        if len(multi_idxs) == 1:
            ax_iter = [axs]
        else:
            ax_iter = axs.flat
        
        for ax in ax_iter:
            ax.set_ylim(instructions['ylim'])
            ax.grid(instructions['showgrid'], linestyle = '--')

    for i, fig in enumerate(figs):

        title = obs_dict[info['obs_name']] + str(title_l[i])

        if instructions['e_title'] != None:

            title = obs_dict[info['obs_name']] + instructions['e_title']

        fig.suptitle(title, size = instructions['textsz'])
        fig.supxlabel(instructions['xlabel'], size = instructions['textsz'])
        fig.supylabel(instructions['ylabel'], size = instructions['textsz'])
        fig.legend(loc = 'center right', title = instructions['legend_t'], fontsize = instructions['textsz']*0.7)
        
        fig.tight_layout()

        show_figure(fig)

        if show == True:
            plt.show()

        if save == True:

            os.makedirs(sv_path, exist_ok = True)

            fig.savefig(sv_path + str(title_l[i]) + '.png', dpi = 300)

        plt.close()

    return
