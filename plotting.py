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
              'PO': ['Parieto-Occipital'],
              'F': ['Frontal'],
              'CFPO': ['Frontal (c)', 'Parieto-Occipital (c)'],
              'TEST': 'TEST'}

obs_dict = {'corrsum': '$C(m,r)$ ',
            'correxp': '$\\nu(m,r)$ ',
            'idim': '$D_{2}(m)$ ',
            'llyap': '$\\lambda(m)$ '}

cond_dict = {'S__': 'Conscious',
             'S_1': 'Unconscious'}


### PLOTTING WRAPPERS ###

def show_figure(fig):

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    return

def plot_1dfunction(OBS: np.ndarray, E_OBS: np.ndarray, X: list, label: list, alpha_m: float, grid: list, figsize: list):

    # Scalar value 4-dimensional array

    # Axis 0 = Subjects
    # Axis 1 = Legend
    # Axis 3 = X value

    obs = OBS[:,:,:]
    e_obs = E_OBS[:,:,:]

    cmap = cm.viridis

    norm = mcolors.Normalize(vmin = 0, vmax = len(label) - 1)

    fig, axs = plt.subplots(grid[0], grid[1], figsize = figsize, sharex = True, sharey = True)

    if grid[0]*grid[1] == 1:
        ax_iter = [axs]
    else:
        ax_iter = axs.flat

    for j, ax in enumerate(ax_iter):

        for c in range(0,obs.shape[1]):

            color = cmap(norm(c))

            if j == 0:
                ax.plot(X, obs[j,c,:], color = color, alpha = 1*alpha_m, label = label[c])
            else:
                ax.plot(X, obs[j,c,:], color = color, alpha = 1*alpha_m)
            
            ax.fill_between(X, obs[j,c,:]-e_obs[j,c,:], obs[j,c,:]+e_obs[j,c,:], alpha = 0.2*alpha_m)

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

    labels = [variables['subjects'],variables['conditions'],variables['pois'],variables['embeddings']]

    # Add extra dimension for consistency
    if len(OBS.shape) == 4:
        
        OBS = np.expand_dims(OBS, axis = 4)
        E_OBS = np.expand_dims(E_OBS, axis = 4)
        
    # Apply instructions for averaging
    if instructions['avg'] != 'none':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            OBS = np.nanmean(OBS, axis = avg[instructions['avg']])
            E_OBS = np.nanmean(E_OBS, axis = avg[instructions['avg']])
            
            OBS = np.expand_dims(OBS, axis = avg[instructions['avg']])
            E_OBS = np.expand_dims(E_OBS, axis = avg[instructions['avg']])

    # Initzialize list for array rearranging
    rearrange = [0,0,0,0,0]

    # Check figures
    if instructions['figure'] == 'pois':

        rearrange[1] = 2

        if clst == True:
            title_l = clust_dict[info['clust_lb']]
        else:
            title_l = labels[2]

    elif instructions['figure'] == 'conditions':

        rearrange[1] = 1

        title_l = [cond_dict[i] for i in labels[1]]

    #elif instructions['figures'] == 'conditions_pois': 

    # Selet multiplot axis
    if instructions['multiplot'] == 'subjects':

        rearrange[2] = 0

        plot_l = labels[0]

    if instructions['legend'] == 'conditions':

        rearrange[3] = 1

        legend_l = [cond_dict[i] for i in labels[1]]

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

    # Rearrange arrays
    obs = np.permute_dims(OBS, rearrange)
    e_obs = np.permute_dims(E_OBS, rearrange)
    
    # Reduce extra axis
    if obs.shape[0] > 1:
        if instructions['reduce_method'] == 'product':
            
            obs = [i for ob in obs for i in ob]
            e_obs = [i for e_ob in e_obs for i in e_ob]

            title_l = [str(i) + ' ' + str(j) for i in labels[rearrange[0]] for j in title_l]

    else:

        obs = obs[0]
        e_obs = e_obs[0]

    print(np.asarray(obs).shape)

    # Create iteration around figures
    OBS = [i for i in obs]
    E_OBS = [i for i in e_obs]

    # Initzialize lists for figures and axes
    figs = []
    axes = []
    for i in range(0, len(obs)):
    
        # Initialize figures
        fig, axis = plot_1dfunction(OBS = OBS[i], E_OBS = E_OBS[i], X = X, label = legend_l, alpha_m = instructions['alpha_m'], grid = instructions['grid'], figsize = instructions['figsize'])

        figs.append(fig)
        axes.append(axis)

    # Cycle around axis and figures to add informations

    grid = instructions['grid']
    # Check if we have multiple axes and initialize proper iterable
    for axs in axes:
        if grid[0]*grid[1] == 1:
            ax_iter = [axs]
        else:
            ax_iter = axs.flat
        
        for ax in ax_iter:
            ax.set_ylim(instructions['ylim'])
            ax.grid(instructions['showgrid'], linestyle = '--')

    for i, fig in enumerate(figs):

        title = obs_dict[info['obs_name']] + str(title_l[i])

        fig.suptitle(title, size = instructions['textsz'])
        fig.supxlabel(instructions['xlabel'], size = instructions['textsz'])
        fig.supylabel(instructions['ylabel'], size = instructions['textsz'])
        fig.legend(loc = 'center right', title = instructions['title_l'], fontsize = instructions['textsz']*0.7)
        
        fig.tight_layout()

        show_figure(fig)

        if show == True:
            plt.show()

        if save == True:

            os.makedirs(sv_path, exist_ok = True)

            fig.savefig(sv_path + str(title_l[i]) + '.png', dpi = 300)

        plt.close()

    return
