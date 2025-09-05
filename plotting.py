# Usual suspects
import os
import json

import numpy as np

import matplotlib.pyplot as plt

# Utility function for observables directories and data
from core import obs_path, obs_data

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
              'CPOF': ['Parieto-Occipital (c)','Frontal (c)']}

obs_dict = {'idim': '$D_{2}(m)$ ',
            'llyap': '$\\lambda(m)$ '}

cond_dict = {'S__': 'Conscious',
             'S_1': 'Unconscious'}


### PLOTTING WRAPPERS ###

def plot_function_subject(OBS: np.ndarray, E_OBS: np.ndarray, X: list, grid: list, figsize: list):

    # Scalar value 4-dimensional array

    # Axis 0 = Subjects
    # Axis 1 = Conditions
    # Axis 2 = Clusters
    # Axis 3 = m: Embedding dimension

    # Initialize list of pyplot objects to handle to main function
    figs = []
    axes = []

    rng = OBS.shape[2]
    print(rng)

    for idx in range (0,rng):
            OBS = OBS[:,:,idx,:]
            E_OBS = E_OBS[:,:,idx,:]

            fig, axs = plt.subplots(grid[0], grid[1], figsize = figsize)

            if grid[0]*grid[1] == 1:
                ax_iter = [axs]
            else:
                ax_iter = axs.flat

            for j, ax in enumerate(ax_iter):

                if j == 0:
                    ax.plot(X, OBS[j,0,:])#, label = label[0])
                    ax.plot(X, OBS[j,1,:])#, label = label[1])
                else:
                    ax.plot(X, OBS[j,0,:])
                    ax.plot(X, OBS[j,1,:])
                
                ax.fill_between(X, OBS[j,0,:]-E_OBS[j,0,:], OBS[j,0,:]+E_OBS[j,0,:], alpha = 0.5)
                ax.fill_between(X, OBS[j,1,:]-E_OBS[j,1,:], OBS[j,1,:]+E_OBS[j,1,:], alpha = 0.5)

            figs.append(fig)
            axes.append(axs)

            #plt.close()

    return figs, axes


# Main wrapper for function over pois
def plot_observable(info: dict, instructions: dict, save = False, verbose = True):

    # Legend

    # Info
    #
    #'exp_name'
    #'avg_trials'
    #'obs_name'
    #'res_lb'
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

    # Value of the array is y(x) or e_y(x)

    # Get observable file path
    # Get relevant paths
    path = obs_path(
                       exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       res_lb = info['res_lb'],
                       calc_lb = info['calc_lb']

                       )

    # Load results for specific observable
    OBS, E_OBS, X, variables = obs_data(path = path, obs_name = info['obs_name'])

    if verbose == True:
        print(variables)

    clst = variables['clustered']

    conditions = variables['conditions']

    # Apply instructions
    if instructions['avg'] != 'none':
        OBS = OBS.mean(axis = avg[instructions['avg']])
        E_OBS = E_OBS.mean(axis = avg[instructions['avg']])

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

    # Swap dimensions for different compatring in the same picture
    if instructions['confront'] == 'clusters':

        OBS = np.swapaxes(OBS,1,2)
        E_OBS = np.swapaxes(E_OBS,1,2)

        title_l = [cond_dict[c] for c in conditions]
        legend_l = clust_dict[info['res_lb']]

        if clst == False or avg[instructions['avg']] != None:
            print('Cluster confrontation of non clustered data or averaged data')
            return

    elif instructions['confront'] == 'conditions':

        title_l = clust_dict[info['res_lb']]
        legend_l = [cond_dict[c] for c in conditions]
        
    # Initialize figures
    figs, axes = plot_function_subject(OBS = OBS, E_OBS = E_OBS, X = X, grid = grid, figsize = instructions['figsize'])

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
        fig.legend(legend_l, loc = 'lower center')
        if save == True:
            sv_path = maind[info['exp_name']]['directories']['pics']
            plt.savefig(sv_path + instructions['sv_name'] + str(i) + '.png', dpi = 300)

    return
