# Usual suspects
import os
import json
import warnings

from tqdm import tqdm

from pprint import pp

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as mcolors

# Utility functions for directories and data
from core import pics_path, obs_path, loadresults

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### MULTIPROCESSING PARAMETERS ###

workers = 4
chunksize = 1

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
              'F': ['Fp1', 'Fp2', 'Fpz'],
              'CFPO': ['Fp1','Fp2','Fpz','PO3','PO4','Oz'],
              'znoisefree': ['Lorenz'],
              'm_znoisefree': ['Lorenz (m)'],
              'm_znoisefree_dense': ['Lorenz (m)'],
              'gnoise': ['Gaussian Noise'],
              'm_gnoise': ['Gaussian Noise (m)'],
              }

obs_dict = {
            'epochs': 'Epoch Time Series ',
            'spectrum': 'Epoch Frequency Spectrum ',
            'delay': '$\\tau$',
            'separation': 'Spacetime Separation ',
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
             'S__1': 'C L Self',
             'S__2': 'C L Other',
             'S__3': 'C R Self',
             'S__4': 'C R Other',
             'S_11': 'U L Self',
             'S_12': 'U L Other',
             'S_13': 'U R Self',
             'S_14': 'U R Other',
             'lorenz': 'Lorenz',
             'noise': 'Noise'
             }

basic_instructions = {
                     'avg': 'none',
                     'dim_m': 1,
                     'reduce_multi': None,
                     'reduce_legend': None,#[i for i in range(0,10)],
                     'colormap': cm.viridis,
                     'markersize': 10,
                     'linewidth': 2,
                     'alpha_m': 0.6,
                     'grid': (6,6),
                     'showgrid': True,
                     'figsize': (16,11),
                     'textsz': 25,
                     'e_title': None,#'Lorenz Attractor (w/o embedding normalization)'
                     'legend_s': True,
                     'legend_loc': 'lower left',
                     'X_transform': None
                      }

epochs_instructions = {
                        'figure': 'pois',
                        'multiplot': 'subjects',
                        'legend': 'conditions',
                        'isolines': None,
                        'axis': 't',
                        'reduce_method': 'trivial',
                        'linewidth': 1,
                        'alpha_m': 0.8,
                        'ylabel': '$ERPs$',
                        'xlim': (None,None),
                        'ylim': (None,None),
                        'style': 'curve',
                        'legend_t': 'Condition',
                        'colormap': cm.Set2,
                        }

spectrum_instructions = {
                        'figure': 'pois',
                        'multiplot': 'subjects',
                        'legend': 'conditions',
                        'isolines': None,
                        'axis': 'freqs',
                        'reduce_method': 'trivial',
                        'linewidth': 1,
                        'alpha_m': 0.8,
                        'ylabel': 'I(f) [dB?]',
                        'xlim': (0,None),
                        'ylim': (None,None),
                        'style': 'curve',
                        'legend_t': 'Condition',
                        'colormap': cm.Set2,
                        }

delay_instructions = {
                        'figure': 'one',
                        'multiplot': 'subjects',
                        'legend': 'conditions',
                        'isolines': None,
                        'axis': 'pois',
                        'reduce_method': 'trivial',
                        'ylabel': '$\\tau$',
                        'xlim': (0,None),
                        'ylim': (12,45),
                        'style': 'marker',
                        'legend_t': 'Condition',
                        }

separation_instructions = {
                        'figure': 'pois',
                        'multiplot': 'subjects',
                        'legend': 'embeddings',
                        'isolines': 'percentiles',
                        'axis': 'dt',
                        'reduce_method': 'product',
                        'ylabel': '$S_{m}(\\Delta_{ij} | \\left|i - j\\right| = \\delta t)$',
                        'xlim': (0,None),
                        'ylim': (0,None),
                        'style': 'curve',
                        'legend_t': 'Embedding\ndimension',
                        }

correxp_instructions = {
                        'figure': 'pois',
                        'multiplot': 'subjects',
                        'legend': 'embeddings',
                        'isolines': None,
                        'axis': 'log_r',
                        'reduce_method': 'product',
                        'ylabel': '$\\nu_{m}(r)$',
                        'xlim': (None,None),
                        'ylim': (0,6),
                        'style': 'curve',
                        'legend_t': 'Embedding\ndimension',
                        }

peaks_instructions = {
                      'figure': 'conditions',
                      'multiplot': 'subjects',  
                      'legend': 'embeddings',
                      'isolines': None,
                      'axis': 'pois',
                      'reduce_method': 'trivial',
                      'ylabel': '$\\nu_{max}$',
                      'xlim': (None,None),
                      'ylim': (1,4),
                      'style': 'marker',
                      'legend_t': 'Embedding\ndimension',
                     }

plateaus_instructions = {
                      'figure': 'conditions',
                      'multiplot': 'subjects',  
                      'legend': 'embeddings',
                      'isolines': None,
                      'axis': 'pois',
                      'reduce_method': 'trivial',
                      'ylabel': '$\\nu_{p}$',
                      'xlim': (None,None),
                      'ylim': (0.8,2.1),
                      'style': 'marker',
                      'legend_t': 'Embedding\ndimension',
                     }

obs_instructions = {
                    'epochs': epochs_instructions,
                    'spectrum': spectrum_instructions,
                    'delay': delay_instructions,
                    'separation': separation_instructions,
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
    instructions = basic_instructions.copy()

    # Obs instructions get priority over basic instructions
    for key in obs_instructions[info['obs_name']].keys():

        instructions[key] = obs_instructions[info['obs_name']][key]

    # Extra instructions get priority over any instructions
    if extra_instructions != None:

        for key in extra_instructions.keys():

            instructions[key] = extra_instructions[key]

    # Get save path
    instructions['sv_path'] = pics_path(
                        exp_name = info['exp_name'],
                       avg_trials = info['avg_trials'],
                       obs_name = info['obs_name'],
                       clust_lb = info['clust_lb'],
                       calc_lb = info['calc_lb']
                       ) + instructions['avg'] + '/'

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
        print(f'### {info['obs_name']}.py script parameters ###\n')
        pp(variables, depth = 1, width = 60)
        print('')

    clst = variables['clustered']

    conditions = variables['conditions']

    if info['obs_name'] in ['corrsum','correxp','peaks','plateaus']:

        # Initzialize list of labels for data
        labels = [variables['subjects'],[cond_dict[i] for i in variables['conditions']],variables['pois'],variables['embeddings'],[instructions['e_title']]]

        # Initzialize list for array rearranging
        rearrange = [0,0,0,0,0]

    elif info['obs_name'] in ['separation']:

        # Initzialize list of labels for data
        labels = [variables['subjects'],[cond_dict[i] for i in variables['conditions']],variables['pois'],variables['embeddings'],variables['percentiles'],[instructions['e_title']]]

        # Initzialize list for array rearranging
        rearrange = [0,0,0,0,0,0]

    elif info['obs_name'] in ['epochs', 'spectrum', 'delay']: 

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

    if instructions['isolines'] == 'percentiles':

        rearrange[-2] = 4

    if instructions['legend'] == 'conditions':

        rearrange[3] = 1

        legend_l = labels[1]

    elif instructions['legend'] == 'embeddings':

        rearrange[3] = 3

        legend_l = labels[3]

    if instructions['axis'] == 'embeddings':

        rearrange[-1] = 3

        x = X[0]

        instructions['xlabel'] = '$m$'

    elif instructions['axis'] == 'log_r':

        rearrange[-1] = 4

        x = X[1]

        instructions['xlabel'] = '$\\log(r)$'

    elif instructions['axis'] == 'dt':

        rearrange[-1] = 5

        x = X[1]

        instructions['xlabel'] = '$\\delta t$'

    elif instructions['axis'] == 't':

        rearrange[-1] = 3

        x = X[0]

        instructions['xlabel'] = '$t$'

    elif instructions['axis'] == 'freqs':

        rearrange[-1] = 3

        x = X[0]

        instructions['xlabel'] = 'f [Hz]'

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

    if info['obs_name'] == 'spectrum':

        o = obs.copy()

        obs = np.log10(o)
        e_obs = e_obs/(np.log(10)*o)

        del o

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
        'x': x,
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

                    count = 0
                    for k in range(0,len(x_full)):

                        if k >= l_b and k < u_b:

                            obs[i,j,k] = o
                            e_obs[i,j,k] = e_o
                            count += 1
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

    first = True
    for j, ax in zip(multi_idxs, ax_iter):

        for i, c in enumerate(label_idxs):

            if type(colormap) != str:

                if cmap == cm.Set2 or cmap == cm.tab20c:

                    color = cmap(i)

                else:

                    color = cmap(norm(i))

            else:

                color = colormap

            if len(obs[j,c].shape) == 1:

                o = [obs[j,c]]
                e_o = [e_obs[j,c]]

            else:

                o = [obs[j,c,i] for i in range(0,obs.shape[2])]
                e_o = [e_obs[j,c,i] for i in range(0,e_obs.shape[2])]

            second = True
            for f, e_f in zip(o, e_o):

                if style == 'curve':
                
                    if first == True and second == True and legend_s == True:
                        ax.plot(x_full, f, '-', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m, label = label[c])
                    else:
                        ax.plot(x_full, f, '-', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m)

                    ax.fill_between(x_full, f - e_f, f + e_f, color = color, alpha = 0.2*alpha_m)

                if style == 'marker':

                    if first == True and second == True and legend_s == True:
                        ax.errorbar(x_full, f, yerr = e_f, fmt = 'o', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m, label = label[c])
                    else:
                        ax.errorbar(x_full, f, yerr = e_f, fmt = 'o', markersize = markersize, linewidth = linewidth, color = color, alpha = 1*alpha_m)

                second = False

        first = False

    plt.close()

    return fig, axs

# Define iterable function for multiprocessing
def it_plot_1d_function(iterable: list):

    l_dict = iterable[0]

    instructions = l_dict['instructions']

    fig, axis = plot_1dfunction(OBS = iterable[1], E_OBS = iterable[2], X = iterable[3], multi_idxs = l_dict['multi_idxs'], label = l_dict['legend_l'], label_idxs = l_dict['label_idxs'],
                                alpha_m = instructions['alpha_m'], grid = instructions['grid'], figsize = instructions['figsize'],
                                style = instructions['style'], colormap = instructions['colormap'],
                                markersize = instructions['markersize'], linewidth = instructions['linewidth'], legend_s = instructions['legend_s'],
                                fig = iterable[4], axs = iterable[5])

    return fig, axis

def make_figures(info: dict, instructions: dict, verbose: bool, figs = None, axes = None):

    # Get data and transform it for adequate plotting
    OBS, E_OBS, x_list, l_dict = transform_data(info = info, instructions = instructions, verbose = verbose)

    # Extract new instructions
    #instructions = l_dict['instructions']

    # Create dummy list of figs and axes
    if figs == None:

        figs = [None for i in range(0,len(OBS))]
        axes = [None for i in range(0,len(OBS))]

    # Create iterable
    iterable = [[l_dict, OBS[i], E_OBS[i], x_list[i], figs[i], axes[i]] for i in range(0,len(OBS))]

    # Launch Pool multiprocessing
    from multiprocessing import Pool
    with Pool(workers) as p:
        
        figsandaxes = list(tqdm(p.imap(it_plot_1d_function, iterable), #chunksize = chunksize),
                       desc = 'Graphics in progress',
                       unit = 'pic',
                       total = len(iterable),
                       leave = False,
                       dynamic_ncols = True,
                       disable = not verbose)
                        )

    figs_ = []
    axes_ = []

    for fa in figsandaxes:

        figs_.append(fa[0])
        axes_.append(fa[1])

    figs = figs_
    axes = axes_

    '''
    # Old single process method (Can get really slow)
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
    '''
    
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

            if info['obs_name'] == 'spectrum':
                ax.set_xscale('log')

            ylocs = ax.get_yticks()
            #ylabels = [f'{yloc: 06f}' for yloc in ylocs]
            ax.set_yticks(ticks = ylocs, minor = False)# fontsize = instructions['textsz']/2)

            xlocs = ax.get_xticks()
            #xlabels = [f'{xloc}' for xloc in xlocs]
            ax.set_xticks(ticks = xlocs, minor = False)# fontsize = instructions['textsz']/2)

            if instructions['showgrid'] != False:
                ax.grid(instructions['showgrid'], linestyle = '--')

            if instructions['axis'] == 'pois':
                ax.set_xticks(ticks = l_dict['x'], labels = l_dict['labels'][2], rotation = 90, fontsize = instructions['textsz']/2)

            ax.set_xlim((l_dict['x'][0],l_dict['x'][-1]))
            #ax.set_xlim(instructions['xlim'])

            ax.tick_params(axis = 'both', which = 'major', labelsize = instructions['textsz']/2)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = instructions['textsz']/2)

    for i, fig in enumerate(figs):

        title = obs_dict[info['obs_name']] + str(l_dict['title_l'][i])

        if instructions['e_title'] != None:

            title = obs_dict[info['obs_name']] + instructions['e_title']

        fig.suptitle(title, size = instructions['textsz'])
        fig.supxlabel(instructions['xlabel'], size = instructions['textsz'])
        fig.supylabel(instructions['ylabel'], size = instructions['textsz'])

        if instructions['legend_s'] == True:
            fig.legend(loc = instructions['legend_loc'], title = instructions['legend_t'],title_fontsize = instructions['textsz']/2, fontsize = instructions['textsz']/2)
        
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
