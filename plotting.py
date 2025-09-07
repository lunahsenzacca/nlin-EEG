# Usual suspects
import os
import json

import numpy as np

import matplotlib.pyplot as plt

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
              'PO': ['Parieto-Occipital'],
              'F': ['Frontal'],
              'CPOF': ['Parieto-Occipital (c)','Frontal (c)'],
              'TEST': 'TEST'}

obs_dict = {'idim': '$D_{2}(m)$ ',
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

def plot_function_subject(OBS: np.ndarray, E_OBS: np.ndarray, X: list, label: list, grid: list, figsize: list):

    # Scalar value 4-dimensional array

    # Axis 0 = Subjects
    # Axis 1 = Conditions
    # Axis 2 = Clusters
    # Axis 3 = m: Embedding dimension

    # Initialize list of pyplot objects to handle to main function
    figs = []
    axes = []

    rng = OBS.shape[2]

    for idx in range (0,rng):
            obs = OBS[:,:,idx,:]
            e_obs = E_OBS[:,:,idx,:]

            fig, axs = plt.subplots(grid[0], grid[1], figsize = figsize)

            if grid[0]*grid[1] == 1:
                ax_iter = [axs]
            else:
                ax_iter = axs.flat

            for j, ax in enumerate(ax_iter):

                for c in range(0,obs.shape[1]):

                    if j == 0:
                        ax.plot(X, obs[j,c,:], label = label[c])

                    else:
                        ax.plot(X, obs[j,c,:])
                    
                    ax.fill_between(X, obs[j,c,:]-e_obs[j,c,:], obs[j,c,:]+e_obs[j,c,:], alpha = 0.5)

            figs.append(fig)
            axes.append(axs)

            plt.close()

    return figs, axes


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

    # Scalar value 4-dimensional array

    # Axis 0 = Subjects
    # Axis 1 = Conditions
    # Axis 2 = Electrodes
    # Axis 3 = m: x
    # Axis 4 = Value and Error (if avg_trials == False)

    # Value of the array is y(x) or e_y(x)

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
    OBS, E_OBS, X, variables = obs_data(obs_path = path, obs_name = info['obs_name'])

    if verbose == True:
        print(variables)

    clst = variables['clustered']

    conditions = variables['conditions']

    # Apply instructions
    if instructions['avg'] != 'none':
        OBS = OBS.mean(axis = avg[instructions['avg']])
        E_OBS = E_OBS.mean(axis = avg[instructions['avg']])

    else:
        clust_dict[info['clust_lb']] = variables['pois']
        print('No plot average was selected, multiple plots will be printed')

    grid = instructions['grid']

    if instructions['avg'] == 'pois':
        OBS = OBS[:,:,np.newaxis,:]
        E_OBS = E_OBS[:,:,np.newaxis,:]
        if clst == True:
            print('Instructions for pois average of clustered data, not the intended use')
            return

    elif instructions['avg'] == 'sub_pois':
        OBS = OBS[np.newaxis,:,np.newaxis,:]
        E_OBS = E_OBS[np.newaxis,:,np.newaxis,:]
        if clst == True:
            print('Instructions for pois average of clustered data, not the intended use')
            return

        # Set grid to trivial
        grid = (1,1)

    elif instructions['avg'] == 'sub':
        OBS = OBS[np.newaxis,:,:,:]
        E_OBS = E_OBS[np.newaxis,:,:,:]
        if clst == False:
            print('Multiplot over pois not yet implemented')
            return

        else:
            # Set grid to trivial
            grid = (1,1)


    # Swap dimensions for different compatring in the same picture
    if instructions['confront'] == 'clusters':

        OBS = np.swapaxes(OBS,1,2)
        E_OBS = np.swapaxes(E_OBS,1,2)

        title_l = [cond_dict[c] for c in conditions]
        legend_l = clust_dict[info['clust_lb']]

        if clst == False and instructions['avg'] != 'none':
            print('Cluster confrontation of non clustered data or averaged data')
            return

    elif instructions['confront'] == 'conditions':

        title_l = clust_dict[info['clust_lb']]
        legend_l = [cond_dict[c] for c in conditions]
        
    # Initialize figures
    figs, axes = plot_function_subject(OBS = OBS, E_OBS = E_OBS, X = X, label = legend_l, grid = grid, figsize = instructions['figsize'])

    # Cycle around axis and figures to add informations

    # Check if we have multiple axes and initialize proper iterable
    for axs in axes:
        if grid[0]*grid[1] == 1:
            ax_iter = [axs]
        else:
            ax_iter = axs.flat
        
        for ax in ax_iter:
            ax.set_ylim(instructions['ylim'])

    for i, fig in enumerate(figs):

        title = obs_dict[info['obs_name']] + title_l[i]

        fig.suptitle(title, size = instructions['titlesz'])
        fig.legend(loc = 'lower center')
        
        show_figure(fig)

        if show == True:
            plt.show()

        if save == True:

            os.makedirs(sv_path, exist_ok = True)

            fig.savefig(sv_path + title_l[i] + '.png', dpi = 300)

        plt.close()

    return
