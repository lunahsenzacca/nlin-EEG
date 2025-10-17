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
            'plateaus': '$\\nu_{p}$ ',
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
                     'xlim': (None,None),
                     'e_title': None,#'Lorenz Attractor (w/o embedding normalization)'
                     'colormap': cm.viridis,
                     'legend_s': True,
                     'legend_loc': 'lower left',
                     'X_transform': None
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
                        'ylim': (0,6),
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

plateaus_instructions = {
                      'figure': 'conditions',
                      'multiplot': 'subjects',  
                      'legend': 'embeddings',
                      'axis': 'pois',
                      'avg': 'none',
                      'reduce_method': 'trivial',
                      'ylabel': '$\\nu_{p}$',
                      'ylim': (0.8,2.1),
                      'style': 'marker',
                      'legend_t': 'Embedding\ndimension',
                     }

obs_instructions = {
                    'delay': delay_instructions,
                    'correxp': correxp_instructions,
                    'peaks': peaks_instructions,
                    'plateaus': plateaus_instructions
                   }

### MAIN PLOTTING WRAPPER ###

# Simple plotting of one observable
def simple_plot(info: dict, extra_instructions = None, show = True, save = False, verbose = True):

    # Compile instructions dictionary
    instructions = make_instructions(info = info, extra_instructions = extra_instructions)

    # Make figures with appropriate data
    figs, axes, l_dict = make_figures(info = info, instructions = instructions, verbose = verbose)

    # Add axis labels and legend
    figs, axes = set_figures(figs = figs, axes = axes, l_dict = l_dict)

    show_figures(figs = figs, l_dict = l_dict, show = show, save = save)

    return

# Layered plotting of two observables
def double_plot(info_1: dict, info_2: dict, extra_instructions_1 = None, extra_instructions_2 = None, show = True, save = False, verbose = True):

    # Compile instructions dictionary
    instructions_1 = make_instructions(info = info_1, extra_instructions = extra_instructions_1)
    instructions_2 = make_instructions(info = info_2, extra_instructions = extra_instructions_2)

    # Replicate first instructions for proper data trransformation
    instructions_2['figure'] = instructions_1['figure']
    instructions_2['multiplot'] = instructions_1['multiplot']
    instructions_2['legend'] = instructions_1['legend']
    instructions_2['axis'] = instructions_1['axis']
    instructions_2['avg'] = instructions_1['avg']
    instructions_2['reduce_method'] = instructions_1['reduce_method']

    instructions_2['ylim'] = instructions_1['ylim']

    # Make figures with appropriate data
    figs, axes, l_dict_1 = make_figures(info = info_1, instructions = instructions_1, verbose = verbose)
    figs, axes, l_dict_2 = make_figures(info = info_2, instructions = instructions_2, verbose = verbose, figs = figs, axes = axes)

    # Add axis labels and legend
    figs, axes = set_figures(figs = figs, axes = axes, l_dict = l_dict_2)

    show_figures(figs = figs, l_dict = l_dict_2, show = show, save = save)

    return

### MODULES ###

def show_figure(fig):

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    return

# Compile instructions dictionary with standard options and extra overrides
def make_instructions(info: dict, extra_instructions: dict):
    # Construct standard instructions arrays
    instructions = basic_instructions | obs_instructions[info['obs_name']]

    # Get save path
    instructions['sv_path'] = pics_path(
                        exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       ) + instructions['avg'] + '/'

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

    return instructions

def transform_data(info: dict, instructions: dict, verbose: bool):

    # Get relevant paths
    path = obs_path(
                       exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       )

    # Load results for specific observable
    results, X, variables = loadresults(obs_path = path, obs_name = info['obs_name'], X_transform = instructions['X_transform'])

    if verbose == True:
        print(variables)

    clst = variables['clustered']

    conditions = variables['conditions']

    if info['obs_name'] in ['corrsum','correxp','peaks','plateaus']:

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

        if instructions['X_transform'] != None:

            X_results = np.asarray(X[instructions['X_transform']])

            X_results = X_results[:,:,0,0]

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

        x = X[0]

        instructions['xlabel'] = '$m$'

    elif instructions['axis'] == 'log_r':

        rearrange[-1] = 4

        x = X[1]

        instructions['xlabel'] = '$\\log(r)$'

    elif instructions['axis'] == 'pois':

        rearrange[-1] = 2

        x = np.asarray([i for i in range(0,len(labels[2]))])

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

    if instructions['X_transform'] != None:

        x_res = np.permute_dims(X_results, rearrange)
    
    # Reduce extra axis
    if instructions['reduce_method'] == 'product':

        title_l = [str(i) + '; ' + str(j) for i in labels[rearrange[0]] for j in title_l]

    obs = [i for ob in obs for i in ob]
    e_obs = [i for e_ob in e_obs for i in e_ob]

    if instructions['X_transform'] != None:

        x_res = [i for res in x_res for i in res]

        x_ = x_res

    else:

        x_ = [None for i in range(0,len(obs))]

    # Create iteration around figures
    OBS = [i for i in obs]
    E_OBS = [i for i in e_obs]

    x_list = [[x,i] for i in x_]

    # Create dictionary for labels titles and reduced plotting
    l_dict = {
        'info': info,
        'instructions': instructions,
        'labels': labels,
        'title_l': title_l,
        'plot_l': plot_l,
        'legend_l': legend_l,
        'label_idxs': label_idxs,
        'multi_idxs': multi_idxs,
        'x': x
    }

    return OBS, E_OBS, x_list, l_dict

def plot_1dfunction(OBS: np.ndarray, E_OBS: np.ndarray, X: list, multi_idxs: list, label: list, label_idxs: list, alpha_m: float, colormap, grid: list, figsize: list, style: str, markersize: str, linewidth: str, legend_s: bool, fig = None, axs = None):

    # Axis 0 = Multiplot
    # Axis 1 = Legend
    # Axis 2 = X value

    obs = OBS[:,:,:]
    e_obs = E_OBS[:,:,:]

    # Fixed x
    x_full = X[0]

    # Ranges
    x_rng = X[1]

    if x_rng is not None:

        obs = np.zeros([*OBS.shape[:-1],len(x_full)])
        e_obs = np.zeros([*E_OBS.shape[:-1],len(x_full)])

        for i in range(0,x_rng.shape[0]):
            for j in range(0,x_rng.shape[1]):

                l_b = x_rng[i,j,0]
                u_b = x_rng[i,j,1]

                if np.isnan(l_b) == False:

                    o = OBS[i,j,0]
                    e_o = E_OBS[i,j,0]

                    for k in range(0,len(x_full)):

                        if k >= l_b and k < u_b:

                            obs[i,j,k] = o
                            e_obs[i,j,k] = e_o

                        else:
                            
                            obs[i,j,k] = np.nan
                            e_obs[i,j,k] = np.nan

                else:

                    for k in range(0,x_rng.shape[2]):

                        obs[i,j,k] = np.nan
                        e_obs[i,j,k] = np.nan

    if type(colormap) != str:

        cmap = colormap

        norm = mcolors.Normalize(vmin = 0, vmax = len(label_idxs) - 1)

    if fig == None:

        fig, axs = plt.subplots(grid[0], grid[1], figsize = figsize, sharex = True, sharey = True)

    if len(multi_idxs) == 1 or grid[0]*grid[1] == 1:
        ax_iter = [axs]
    else:
        ax_iter = axs.flat

    n = 0
    for j, ax in zip(multi_idxs, ax_iter):

        for i, c in enumerate(label_idxs):

            if type(colormap) != str:

                color = cmap(norm(i))

            else:

                color = colormap

            if style == 'curve':
            
                if n == 0 and legend_s == True:
                    ax.plot(x_full, obs[j,c,:], '-', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m, label = label[c])
                else:
                    ax.plot(x_full, obs[j,c,:], '-', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m)

                ax.fill_between(x_full, obs[j,c,:]-e_obs[j,c,:], obs[j,c,:]+e_obs[j,c,:], color = color, alpha = 0.4*alpha_m)

            if style == 'marker':

                if n == 0 and legend_s == True:
                    ax.errorbar(x_full, obs[j,c,:], yerr = e_obs[j,c,:], fmt = 'o', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m, label = label[c])
                else:
                    ax.errorbar(x_full, obs[j,c,:], yerr = e_obs[j,c,:], fmt = 'o', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m)

        n += 1

    plt.close()

    return fig, axs

def make_figures(info: dict, instructions: dict, verbose: bool, figs = None, axes = None):

    # Get data and transform it for adequate plotting
    OBS, E_OBS, x_list, l_dict = transform_data(info = info, instructions = instructions, verbose = verbose)

    # Extract new instructions
    instructions = l_dict['instructions']

    # Create dummy list of figs and axes
    if figs == None:

        figs = [None for i in range(0,len(OBS))]
        axes = [None for i in range(0,len(OBS))]

    # Initzialize new lists for figures and axes
    figs_ = []
    axes_ = []
    for i in range(0, len(OBS)):
    
        # Initialize figures
        fig, axis = plot_1dfunction(OBS = OBS[i], E_OBS = E_OBS[i], X = x_list[i], multi_idxs = l_dict['multi_idxs'], label = l_dict['legend_l'], label_idxs = l_dict['label_idxs'],
                                    alpha_m = instructions['alpha_m'], grid = instructions['grid'], figsize = instructions['figsize'],
                                    style = instructions['style'], colormap = instructions['colormap'],
                                    markersize = instructions['markersize'], linewidth = instructions['linewidth'], legend_s = instructions['legend_s'],
                                    fig = figs[i], axs = axes[i])

        figs_.append(fig)
        axes_.append(axis)

    figs = figs_
    axes = axes_

    return figs, axes, l_dict

def set_figures(figs: list, axes: list, l_dict: dict):

    info = l_dict['info']

    instructions = l_dict['instructions']

    # Cycle around axis and figures to add informations
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
                ax.set_xticks(ticks = l_dict['x'], labels = l_dict['labels'][2], rotation = 90, fontsize = instructions['textsz']/2)

            ax.set_xlim(instructions['xlim'])

    for i, fig in enumerate(figs):

        title = obs_dict[info['obs_name']] + str(l_dict['title_l'][i])

        if instructions['e_title'] != None:

            title = obs_dict[info['obs_name']] + instructions['e_title']

        fig.suptitle(title, size = instructions['textsz'])
        fig.supxlabel(instructions['xlabel'], size = instructions['textsz'])
        fig.supylabel(instructions['ylabel'], size = instructions['textsz'])

        if instructions['legend_s'] == True:
            fig.legend(loc = instructions['legend_loc'], title = instructions['legend_t'], fontsize = instructions['textsz']*0.7)
        
        fig.tight_layout()

        show_figure(fig)

        plt.close()

    return figs, axes

def show_figures(figs: list, l_dict: dict, show = True, save = False):

    sv_path = l_dict['instructions']['sv_path']

    for i, fig in enumerate(figs):

        show_figure(fig)

        if show == True:
            plt.show()

        if save == True:

            os.makedirs(sv_path, exist_ok = True)

            fig.savefig(sv_path + str(l_dict['title_l'][i]) + '.png', dpi = 300)

        plt.close()

    return
