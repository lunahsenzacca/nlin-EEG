# Usual suspects
import os
import json
import warnings

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as mcolors

# Utility functions for directories and data
from core import pics_path, obs_path, loadresults

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### Extra DICTIONARIES (Eventually put in init.py)###

avg = {
       'pois': 2,
       'sub_pois': (0,2),
       'sub': 0
       }

clust_dict = {
              'G': ['Global'],
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
              'mCFPOdense':['Fp1','Fp2','Fpz','Frontal','O2','PO4','PO8','Parieto-Occipital','Fronto-Parieto-Occipital System'],
              'test': ['test']
              }

obs_dict = {
            'delay': '$\\tau$',
            'corrsum': '$C_{m}(r)$ ',
            'correxp': '$\\nu_{m}(r)$ ',
            'peaks': '$\\nu_{max}$ ',
            'idim': '$D_{2}(m)$ ',
            'llyap': '$\\lambda(m)$ '
            }

cond_dict = {
             'S__': 'Conscious',
             'S_1': 'Unconscious',
             'lorenz': 'Lorenz',
             'noise': 'Noise'
             }

basic_instructions = {
                     'dim_m': 1,
                     'reduce_multi': None,
                     'reduce_legend': None,#[i for i in range(0,10)],
                     'markersize': 10,
                     'linewidth': 2,
                     'alpha_m': 0.6,
                     'grid': (6,6),
                     'showgrid': True,
                     'figsize': (16,11),
                     'textsz': 25,
                     'e_title': None#'Lorenz Attractor (w/o embedding normalization)'
                      }

delay_instructions = {
                        'figure': 'one',
                        'multiplot': 'subjects',
                        'legend': 'conditions',
                        'axis': 'pois',
                        'avg': 'none',
                        'reduce_method': 'trivial',
                        'ylabel': '$\\tau$',
                        'ylim': (12,45),
                        'style': 'marker',
                        'legend_t': 'Condition',
                        }

correxp_instructions = {
                        'figure': 'pois',
                        'multiplot': 'subjects',
                        'legend': 'embeddings',
                        'axis': 'log_r',
                        'avg': 'none',
                        'reduce_method': 'product',
                        'ylabel': '$\\nu_{m}(r)$',
                        'ylim': (0,7),
                        'style': 'curve',
                        'legend_t': 'Embedding\ndimension',
                        }

peaks_instructions = {
                      'figure': 'conditions',
                      'multiplot': 'subjects',  
                      'legend': 'embeddings',
                      'axis': 'pois',
                      'avg': 'none',
                      'reduce_method': 'trivial',
                      'ylabel': '$\\nu_{max}$',
                      'ylim': (1,4),
                      'style': 'marker',
                      'legend_t': 'Embedding\ndimension',
                     }

obs_instructions = {
                    'delay': delay_instructions,
                    'correxp': correxp_instructions,
                    'peaks': peaks_instructions,
                   }

### PLOTTING WRAPPERS ###

def show_figure(fig):

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    return

def transform_data(info:dict, instructions: dict, verbose: bool):

    # Get relevant paths
    path = obs_path(
                       exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       )

    # Load results for specific observable
    results, X, variables = loadresults(obs_path = path, obs_name = info['obs_name'])

    if verbose == True:
        print(variables)

    clst = variables['clustered']

    conditions = variables['conditions']

    if info['obs_name'] in ['corrsum','correxp','peaks']:

        # Initzialize list of labels for data
        labels = [variables['subjects'],[cond_dict[i] for i in variables['conditions']],variables['pois'],variables['embeddings'],[instructions['e_title']]]

        # Initzialize list for array rearranging
        rearrange = [0,0,0,0,0]

    elif info['obs_name'] in ['delay']: 

        # Initzialize list of labels for data
        labels = [variables['subjects'],[cond_dict[i] for i in variables['conditions']],variables['pois'],[instructions['e_title']]]

        # Initzialize list for array rearranging
        rearrange = [0,0,0,0,0]
    
    if info['avg_trials'] == True:

        results = np.asarray(results)

        OBS = results[:,:,0,0]
        E_OBS = results[:,:,0,1]

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

    # Check figures
    if instructions['figure'] == 'pois':

        rearrange[1] = 2

        title_l = labels[2]

    elif instructions['figure'] == 'conditions':

        rearrange[1] = 1

        title_l = labels[1]

    elif instructions['figure'] == 'one':

        rearrange[1] = 4

        title_l = ['']

    # Selet multiplot axis
    if instructions['multiplot'] == 'subjects':

        rearrange[2] = 0

        plot_l = labels[0]

    if instructions['legend'] == 'conditions':

        rearrange[-2] = 1

        legend_l = labels[1]

    elif instructions['legend'] == 'embeddings':

        rearrange[-2] = 3

        legend_l = labels[3]

    if instructions['axis'] == 'embeddings':

        rearrange[-1] = 3

        X = X[0]

        instructions['xlabel'] = '$m$'

    elif instructions['axis'] == 'log_r':

        rearrange[-1] = 4

        X = X[1]

        instructions['xlabel'] = '$\\log(r)$'

    elif instructions['axis'] == 'pois':

        rearrange[-1] = 2

        X = np.asarray([i for i in range(0,len(labels[2]))])

        instructions['xlabel'] = 'POIs'

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

    if info['obs_name'] in ['delay']:

        OBS = np.expand_dims(OBS, axis = (3,4))
        E_OBS = np.expand_dims(E_OBS, axis = (3,4))

    # Rearrange arrays
    obs = np.permute_dims(OBS, rearrange)
    e_obs = np.permute_dims(E_OBS, rearrange)
    
    # Reduce extra axis
    if instructions['reduce_method'] == 'product':

        title_l = [str(i) + '; ' + str(j) for i in labels[rearrange[0]] for j in title_l]
    '''
    if info['obs_name'] in ['corrsum','correxp','peaks']:

        obs = [i for ob in obs for i in ob]
        e_obs = [i for e_ob in e_obs for i in e_ob]

    '''
    obs = [i for ob in obs for i in ob]
    e_obs = [i for e_ob in e_obs for i in e_ob]

    print(np.asarray(obs).shape)

    # Create dictionary for labels titles and reduced plotting
    l_dict = {
        'instructions': instructions,
        'labels': labels,
        'title_l': title_l,
        'plot_l': plot_l,
        'legend_l': legend_l,
        'label_idxs': label_idxs,
        'multi_idxs': multi_idxs
    }

    return obs, e_obs, X, l_dict

def plot_1dfunction(OBS: np.ndarray, E_OBS: np.ndarray, X: list, multi_idxs: list, label: list, label_idxs: list, alpha_m: float, grid: list, figsize: list, style: str, markersize: str, linewidth: str):

    # Axis 0 = Multiplot
    # Axis 1 = Legend
    # Axis 2 = X value

    obs = OBS[:,:,:]
    e_obs = E_OBS[:,:,:]

    print(obs.shape)

    cmap = cm.viridis

    norm = mcolors.Normalize(vmin = 0, vmax = len(label_idxs) - 1)

    fig, axs = plt.subplots(grid[0], grid[1], figsize = figsize, sharex = True, sharey = True)

    if len(multi_idxs) == 1 or grid[0]*grid[1] == 1:
        ax_iter = [axs]
    else:
        ax_iter = axs.flat

    n = 0
    for j, ax in zip(multi_idxs, ax_iter):

        for i, c in enumerate(label_idxs):

            color = cmap(norm(i))

            if style == 'curve':
            
                if n == 0:
                    ax.plot(X, obs[j,c,:], '-', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m, label = label[c])
                else:
                    ax.plot(X, obs[j,c,:], '-', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m)

                ax.fill_between(X, obs[j,c,:]-e_obs[j,c,:], obs[j,c,:]+e_obs[j,c,:], color = color, alpha = 0.4*alpha_m)

            if style == 'marker':

                if n == 0:
                    ax.errorbar(X, obs[j,c,:], yerr = e_obs[j,c,:], fmt = 'o', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m, label = label[c])
                else:
                    ax.errorbar(X, obs[j,c,:], yerr = e_obs[j,c,:], fmt = 'o', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m)

        n += 1

    plt.close()

    return fig, axs


# Main wrapper for function over pois
def plot_observable(info: dict, extra_instructions = None, show = True, save = False, verbose = True):

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

    # Construct standard instructions arrays
    instructions = basic_instructions | obs_instructions[info['obs_name']]

    # Override standard instructions with provided ones
    if extra_instructions != None:

        for key in extra_instructions.keys():

            instructions[key] = extra_instructions[key]

    # Check for confliction instructions
    check_conflict = [instructions['figure'], instructions['multiplot'], instructions['legend'], instructions['axis']]
    if len(set(check_conflict)) != len(check_conflict):
        print('Conflicting instructions,\'figure\', \'multiplot\', \'legend\' and \'axis\', should all be different')
        return

    # Apply dimension multiplier
    resizable = ['markersize','linewidth','textsz']

    for key in resizable:

        instructions[key] = instructions['dim_m']*instructions[key]

    # Get save path
    sv_path = pics_path(
                        exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       ) + instructions['avg'] + '/'

    # Get data and transform it for adequate plotting
    obs, e_obs, X, l_dict = transform_data(info = info, instructions = instructions, verbose = verbose)

    # Extract new instructions
    instructions = l_dict['instructions']

    # Create iteration around figures
    OBS = [i for i in obs]
    E_OBS = [i for i in e_obs]

    # Initzialize lists for figures and axes
    figs = []
    axes = []
    for i in range(0, len(obs)):
    
        # Initialize figures
        fig, axis = plot_1dfunction(OBS = OBS[i], E_OBS = E_OBS[i], X = X, multi_idxs = l_dict['multi_idxs'], label = l_dict['legend_l'], label_idxs = l_dict['label_idxs'],
                                    alpha_m = instructions['alpha_m'], grid = instructions['grid'], figsize = instructions['figsize'],
                                    style = instructions['style'],
                                    markersize = instructions['markersize'], linewidth = instructions['linewidth'])

        figs.append(fig)
        axes.append(axis)

    # Cycle around axis and figures to add informations

    grid = instructions['grid']
    # Check if we have multiple axes and initialize proper iterable
    for axs in axes:
        if len(l_dict['multi_idxs']) == 1 or instructions['grid'][0]*instructions['grid'][1] == 1:
            ax_iter = [axs]
        else:
            ax_iter = axs.flat
        
        for ax in ax_iter:
            ax.set_ylim(instructions['ylim'])

            ylocs = ax.get_yticks()
            ylabels = [f'{yloc: .1f}' for yloc in ylocs]
            ax.set_yticks(ticks = ylocs, labels = ylabels, fontsize = instructions['textsz']/2)

            xlocs = ax.get_xticks()
            xlabels = [f'{xloc: .1f}' for xloc in xlocs]
            ax.set_xticks(ticks = xlocs, labels = xlabels, fontsize = instructions['textsz']/2)

            if instructions['showgrid'] != False:
                ax.grid(instructions['showgrid'], linestyle = '--')

            if instructions['axis'] == 'pois':
                ax.set_xticks(ticks = X, labels = l_dict['labels'][2], rotation = 90, fontsize = instructions['textsz']/2)

    for i, fig in enumerate(figs):

        title = obs_dict[info['obs_name']] + str(l_dict['title_l'][i])

        if instructions['e_title'] != None:

            title = obs_dict[info['obs_name']] + instructions['e_title']

        fig.suptitle(title, size = instructions['textsz'])
        fig.supxlabel(instructions['xlabel'], size = instructions['textsz'])
        fig.supylabel(instructions['ylabel'], size = instructions['textsz'])
        fig.legend(loc = 'lower left', title = instructions['legend_t'], fontsize = instructions['textsz']*0.7)
        
        fig.tight_layout()

        show_figure(fig)

        if show == True:
            plt.show()

        if save == True:

            os.makedirs(sv_path, exist_ok = True)

            fig.savefig(sv_path + str(l_dict['title_l'][i]) + '.png', dpi = 300)

        plt.close()

    return
