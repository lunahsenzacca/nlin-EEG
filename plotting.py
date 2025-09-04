# Usual suspects
import os
import json

import numpy as np

import matplotlib.pyplot as plt

# Utility function for observables directories
from core import obs_path

# Our Very Big Dictionary
from init import get_maind

maind = get_maind()

### Extra DICTIONARIES (Eventually put in init.py)###

clust_dict = {'G': 'Global (m $\\leq$ 10)',
              'G20': 'Global (m $\\leq$ 20)',
              'PO': 'O2, PO4, PO8',
              'F': 'Fp1, Fp2, Fpz',
              'CPOF': ['O2, PO4, PO8','Fp1, Fp2, Fpz']}


### PLOTTING WRAPPERS ###

# Plot scalar observable in ofg 4-dimensional array
def plot_scalar(exp_name: str, avg_trials: bool, obs_name: str, res_lb: str, avg_method: str, calc_lb = None, verbose = True):

    # Legend

    # Scalar value 4-dimensional array

    # Axis 0 = Subjects
    # Axis 1 = Conditions
    # Axis 2 = Electrodes
    # Axis 3 = m: Embedding dimension

    method_d = {

        'subjects': 0,
        'conditions': 1,
        'pois': 2,
        'embeddings': 3,
        'all': (0,1,2,3)

    }

    if obs_name == 'idim':

        # Get relevant paths
        d2_path = obs_path(exp_name = exp_name, obs_name = 'idim', res_lb = res_lb, calc_lb = calc_lb, avg_trials = avg_trials)

        # Load result variables
        with open(d2_path + 'variables.json', 'r') as f:
            variables = json.load(f)

        # Get input shape
        shape0 = variables['shape0']

        # Get output shape
        shape1 = variables['shape1']

        embs = variables['embeddings']

        clst = variables['clustered']

        avg = variables['avg']

        # Fit parameters
        vmin = variables['vlim'][1]
        vmax = variables['vlim'][0]

        # Load results from idim.py script

        m = np.load(d2_path + 'slopes.npy')
        em = np.load(d2_path + 'errslopes.npy')

        if verbose == True:
            print(variables)

        # Make appropriate labeling
        sv_lb = res_lb

        if avg == True:
            a_lb = 'avg'
        else:
            a_lb = ''

        info_lb = '_' + a_lb + str(vmin) + '_' + str(vmax)

        sv_path = maind[exp_name]['directories']['pics']

        if clst == True:

            print('\nClustered input: \'avg_method\' bypassed, more pictures will be printed.')

            rng = len(clust_dict[res_lb])
            
            for cl_idx in range (0,rng):
                M = m[:,:,cl_idx,:]
                EM = em[:,:,cl_idx,:]

                sv_lb = res_lb + '_ ' + str(cl_idx) + info_lb
                title = '$D_{2}(m)$ [' + clust_dict[res_lb][cl_idx] + ']'

                fig, axs = plt.subplots(6,6, figsize = (12,10))

                for j, ax in enumerate(axs.flat):
                    
                    ax.plot(embs, M[j,0,:], label = 'Conscious')
                    ax.fill_between(embs, M[j,0,:]-EM[j,0,:], M[j,0,:]+EM[j,0,:], alpha = 0.5)

                    ax.plot(embs, M[j,1,:], label = 'Unconscious')
                    ax.fill_between(embs, M[j,1,:]-EM[j,1,:], M[j,1,:]+EM[j,1,:], alpha = 0.5)

                    ax.set_ylim(1.2,3)
                    #ax.set_title(sub_list[j])

                fig.suptitle(title, size = 25)

                plt.savefig(sv_path + sv_lb + '_Dattractor.png', dpi = 300)

                fig.show()

        else:

            M = m.mean(axis = method_d['pois'])
            EM = em.mean(axis = method_d['pois'])

            sv_lb = res_lb + info_lb

            title = '$D_{2}(m)$ [' + clust_dict[res_lb] + ']'

            fig, axs = plt.subplots(6,6, figsize = (12,10))

            for j, ax in enumerate(axs.flat):
                
                if j == 0:
                    ax.plot(embs, M[j,0,:], label = 'Conscious')
                    ax.plot(embs, M[j,1,:], label = 'Unconscious')
                else:
                    ax.plot(embs, M[j,0,:])
                    ax.plot(embs, M[j,1,:])

                ax.fill_between(embs, M[j,0,:]-EM[j,0,:], M[j,0,:]+EM[j,0,:], alpha = 0.5)
                ax.fill_between(embs, M[j,1,:]-EM[j,1,:], M[j,1,:]+EM[j,1,:], alpha = 0.5)

                ax.set_ylim(1.2,2.7)
                #ax.set_title(sub_list[j])

            fig.suptitle(title, size = 25)

            fig.legend(loc = 'lower center')

            plt.savefig(sv_path + sv_lb + '_Dattractor.png', dpi = 300)

            fig.show()

    return
