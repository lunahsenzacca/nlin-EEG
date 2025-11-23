'''
In this file we specify all of the experimental and analysis parameters
including directory management into a single dictionary and specify a 
function for loading it and saving it in a notebook.

The end goal is to have every dataset specific entry here and have
a generic program around it.

The main method of the script saves the dictionary into a .json file
'''

### INITIALIZATION ###

   #vvvvvvvvvvvvvv#

# Main folder where data and results are stored
path = '/home/lunis/Documents/nlin-EEG/'

# Data subfolder
d_path = path + 'data/'

# Results subfolder
r_path = path + 'results/'

# Pictures subfolder
p_path = path + 'pics/'

# Experiments and observables labels (More observables coming)
exp_lb = {
    'bmasking': 'BM',
    'zbmasking': 'ZBM',
    'bmasking_dense': 'BMD',
    'zbmasking_dense': 'ZBMD',
    'bmasking_dense_noICA': 'BMDnoICA',
    'zbmasking_dense_noICA':'ZBMDnoICA',
    'bmasking_dense_highICA': 'BMDhighICA',
    'zbmasking_dense_highICA':'ZBMDhighICA',
    'flash_suppression' : 'FS',
    'lorenz': 'LZ',
    'noise': 'NZ'
}
obs_lb = {
    'evokeds': 'EV',
#    'epochs': 'EP',
    'spectrum': 'SP',
    'delay': 'TAU',
#    'recurrence': 'RP',
    'separation': 'SP',
#    'corrsum': 'CS',
    'correxp': 'CE',
    'peaks': 'PK',
    'plateaus': 'PL',
#    'idim': 'D2',
#    'llyap': 'LLE'

}

def paths(exp_name: str, old = False):

    paths = {'data': d_path + exp_lb[exp_name] + '/',
    'avg_data': path + 'toMNE/avg/' + exp_lb[exp_name] + '/',
    'trl_data': path + 'toMNE/trl/' + exp_lb[exp_name] + '/',
    'subject': [d_path + exp_lb[exp_name] + '/','/'],
    'avg_results': r_path + 'avg/' + exp_lb[exp_name] + '/',
    'trl_results': r_path + 'trl/' + exp_lb[exp_name] + '/',
    'avg_pics': p_path +'avg/' + exp_lb[exp_name] + '/',
    'trl_pics': p_path +'trl/' + exp_lb[exp_name] + '/'}

    if old == True:

        paths['subject'] = [d_path + exp_lb[exp_name] + '/subj','_band_resample/']

    return paths

'''
We summarize every dataset-specific information in a dictionary
with the following keys:

T           = Time point lenght of trials;

f           = Sampling frequency;

montage     = EEG headset montage standard name;

subIDs      = Subject ids found in data folder

pois        = Electrode names used in the experiment;

conditions  = Conditions list or dictionary;

directories = Strings for dataset and results navigation;

k           = Nearest neighbours used for autocorrelation time
              estimation by first minimum of Mutual Information (FMMI);

Rth        = Threshold for False Nearest Neighbour (FNN) for embedding
             dimension estimation;

tau        = Timepoints delay used for time series embedding.
'''

### Backward Masking dataset infos ###

BM_subids = ['001','002','003','004','005','006','007','008','009','010',
            '011','012','013','014','015','016','017','018','019','020',
            '023','024','025','026','028','029','030',
            '031','033','034','035','036','037','038','040',
            '042']

BM_pois = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
           'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Iz',
           'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
           'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3',
           'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5',
           'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
           'Fpz', 'CPz', 'POz', 'Oz']

BM_conditions = {'Conscious':'S__',
                 'Unconcious':'S_1',
                 'C L Self': 'S__1',
                 'C L Other': 'S__2',
                 'C R Self': 'S__3',
                 'C R Other': 'S__4',
                 'U L Self': 'S_11',
                 'U L Other': 'S_12',
                 'U R Self': 'S_13',
                 'U R other': 'S_14',}

BM_conditions_IDs = {'S__': [1,2,3,4],
                     'S_1': [11,12,13,14],
                     'S__1': 1,
                     'S__2': 2,
                     'S__3': 3,
                     'S__4': 4,
                     'S_11': 11,
                     'S_12': 12,
                     'S_13': 13,
                     'S_14': 14,}

BM_paths = paths('bmasking', old = True)

# Summarize in a dictionary
BM_info = {
    'T': 451,
    'f': 500,
    'subIDs': BM_subids,
    'pois': BM_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': BM_paths,
    'tau': 20,
    'window': [-0.2,0.7]
}

# Create similar dictionary for zscored results
ZBM_paths = paths('zbmasking', old = True)

ZBM_info = {
    'T': 451,
    'f': 500,
    'subIDs': BM_subids,
    'pois': BM_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': ZBM_paths,
    'tau': 15,
    'window': [-0.2,0.7]

}

BMD_pois = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2',
            'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'F1', 'F2',
            'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4',
            'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8',
            'PO7', 'PO8', 'Fpz', 'CPz', 'POz', 'Oz']

# Create similar dictionary for new preprocessing
BMD_paths = paths('bmasking_dense')

BMD_info = {
    'T': 901,
    'f': 1000,
    'subIDs': BM_subids,
    'pois': BMD_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': BMD_paths,
    'tau': 35,
    'window': [-0.2,0.7]
}

# Create similar dictionary for new preprocessing
ZBMD_paths = paths('zbmasking_dense')

ZBMD_info = {
    'T': 901,
    'f': 1000,
    'subIDs': BM_subids,
    'pois': BMD_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': ZBMD_paths,
    'tau': 35,
    'window': [-0.2,0.7]
}

# Create similar dictionary for new preprocessing
BMD_noICA_paths = paths('bmasking_dense_noICA')

BMD_noICA_info = {
    'T': 901,
    'f': 1000,
    'subIDs': BM_subids,
    'pois': BMD_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': BMD_noICA_paths,
    'tau': 35,
    'window': [-0.2,0.7]
}

# Create similar dictionary for new preprocessing
ZBMD_noICA_paths = paths('zbmasking_dense_noICA')

ZBMD_noICA_info = {
    'T': 901,
    'f': 1000,
    'subIDs': BM_subids,
    'pois': BMD_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': ZBMD_noICA_paths,
    'tau': 35,
    'window': [-0.2,0.7]
}

# Create similar dictionary for new preprocessing
BMD_highICA_paths = paths('bmasking_dense_highICA')

BMD_highICA_info = {
    'T': 901,
    'f': 1000,
    'subIDs': BM_subids,
    'pois': BMD_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': BMD_highICA_paths,
    'tau': 35,
    'window': [-0.2,0.7]
}

# Create similar dictionary for new preprocessing
ZBMD_highICA_paths = paths('zbmasking_dense_highICA')

ZBMD_highICA_info = {
    'T': 901,
    'f': 1000,
    'subIDs': BM_subids,
    'pois': BMD_pois,
    'conditions': BM_conditions,
    'conditions_IDs': BM_conditions_IDs,
    'directories': ZBMD_highICA_paths,
    'tau': 35,
    'window': [-0.2,0.7]
}


### Continuous Flash Suppresion dataset (At Some Point...) ###

FS_subids = {
}

FS_pois = {
}

FS_conditions = {
}

FS_paths = paths('flash_suppression')

FS_info = {
}

LZ_conditions = {'Lorenz':'lorenz'}

LZ_paths = paths('lorenz')

LZ_info = {
    'T': 1000,
    'pois': [0],
    'subIDs': ['000','100','200'],
    'conditions': LZ_conditions,
    'directories': LZ_paths,
    'tau': 20,
    'window': [0,10] # Placeholder, there is actually a pysically reasonable value from this
}

NZ_conditions = {'Noise':'noise'}

NZ_paths = paths('noise')

NZ_info = {
    'T': 451,
    'pois': [0],
    'subIDs': ['000'],
    'conditions': NZ_conditions,
    'directories': NZ_paths,
    'tau': 20,
    'window': [0,0.9]
}


# Save all of this in a Very Big Dictionary
maind = {
    'path': path,
    'bmasking': BM_info,
    'zbmasking': ZBM_info,
    'bmasking_dense': BMD_info,
    'zbmasking_dense': ZBMD_info,
    'bmasking_dense_noICA': BMD_noICA_info,
    'zbmasking_dense_noICA': ZBMD_noICA_info,
    'bmasking_dense_highICA': BMD_highICA_info,
    'zbmasking_dense_highICA': ZBMD_highICA_info,
    'fsuppression': FS_info,
    'lorenz': LZ_info,
    'noise': NZ_info,
    'exp_lb': exp_lb,
    'obs_lb': obs_lb
}

import json

# Get the dictionary anywhere, additional option to save it in .json file
def get_maind(save = False):

    if save == True:
        with open(path + 'maind.json', 'w') as f:
            json.dump(maind, f)

    return maind

# Main method
if __name__ == '__main__':
    get_maind(save = True)
