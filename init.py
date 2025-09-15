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
    'fsuppression' : 'FS',
    'lorenz': 'LZ',
    'noise': 'NZ'
}
obs_lb = {
    'corrsum': 'CS',
    'correxp': 'CE',
    'idim': 'D2',
    'llyap': 'LLE'

}

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

BM_conditions = {'con_all':'S__',
                 'uncon_all':'S_1',
                 'con_left_self': 'S__1',
                 'con_left_other': 'S__2',
                 'con_right_self': 'S__3',
                 'con_right_other': 'S__4',
                 'uncon_left_self': 'S_11',
                 'uncon_left_other': 'S_12',
                 'uncon_right_self': 'S_13',
                 'uncon_right_other': 'S_14',}

BM_paths = {
    'rw_data': d_path + exp_lb['bmasking'] + '/',
    'avg_data': path + 'evoked/avg/' + exp_lb['bmasking'] + '/',
    'trl_data': path + 'evoked/trl/' + exp_lb['bmasking'] + '/',
    'subject': [d_path + exp_lb['bmasking'] + '/subj','_band_resample/'],
    'ch_info': d_path +  exp_lb['bmasking'] + '/subj001_band_resample/channel.mat',
    'avg_results': r_path + 'avg/' + exp_lb['bmasking'] + '/',
    'trl_results': r_path + 'trl/' + exp_lb['bmasking'] + '/',
    'avg_pics': p_path +'avg/' + exp_lb['bmasking'] + '/',
    'trl_pics': p_path +'trl/' + exp_lb['bmasking'] + '/'
}

# Summarize in a dictionary
BM_info = {
    'T': 451,
    'f': 500,
    'montage': 'standard_1020',
    'subIDs': BM_subids,
    'pois': BM_pois,
    'conditions': BM_conditions,
    'directories': BM_paths,
    'k': 5,
    'Rth': 15,
    'tau': 20,
    'avT': 40
}

# Create similar dictionary for zscored results
ZBM_paths = {
    'rw_data': d_path + exp_lb['zbmasking'] + '/',
    'avg_data': path + 'evoked/avg/' + exp_lb['zbmasking'] + '/',
    'trl_data': path + 'evoked/trl/' + exp_lb['zbmasking'] + '/',
    'subject': [d_path + exp_lb['bmasking'] + '/subj','_band_resample/'],
    'ch_info': d_path +  exp_lb['bmasking'] + '/subj001_band_resample/channel.mat',
    'avg_results': r_path + 'avg/' + exp_lb['zbmasking'] + '/',
    'trl_results': r_path + 'trl/' + exp_lb['zbmasking'] + '/',
    'avg_pics': p_path +'avg/' + exp_lb['zbmasking'] + '/',
    'trl_pics': p_path +'trl/' + exp_lb['zbmasking'] + '/'
}

ZBM_info = {
    'T': 451,
    'f': 500,
    'montage': 'standard_1020',
    'subIDs': BM_subids,
    'pois': BM_pois,
    'conditions': BM_conditions,
    'directories': ZBM_paths,
    'k': 5,
    'Rth': 15,
    'tau': 20,
    'avT': 40
}


### Continuous Flash Suppresion dataset (At Some Point...) ###

FS_subids = {
}

FS_pois = {
}

FS_conditions = {
}

FS_paths = {
    'data': d_path + exp_lb['fsuppression'] + '/',
    'subject': [d_path + exp_lb['fsuppression'] + '/',''],
    'results': r_path + exp_lb['fsuppression'] + '/',
    'pics': p_path + exp_lb['fsuppression'] + '/'
}

FS_info = {
}

LZ_conditions = {'lorenz':'lorenz'}

LZ_paths = {
    'avg_data': path + 'evoked/avg/' + exp_lb['lorenz'] + '/',
    'trl_data': path + 'evoked/trl/' + exp_lb['lorenz'] + '/',
    'avg_results': r_path + 'avg/' + exp_lb['lorenz'] + '/',
    'trl_results': r_path + 'trl/' + exp_lb['lorenz'] + '/',
    'avg_pics': p_path +'avg/' + exp_lb['lorenz'] + '/',
    'trl_pics': p_path +'trl/' + exp_lb['lorenz'] + '/'
}

LZ_info = {
    'pois': [0],
    'subIDs': ['000'],
    'conditions': LZ_conditions,
    'directories': LZ_paths,
    'k': 5,
    'Rth': 15,
    'tau': 20,
    'avT': 55
}

NZ_conditions = {'noise':'noise'}

NZ_paths = {
    'avg_data': path + 'evoked/avg/' + exp_lb['noise'] + '/',
    'trl_data': path + 'evoked/trl/' + exp_lb['noise'] + '/',
    'avg_results': r_path + 'avg/' + exp_lb['noise'] + '/',
    'trl_results': r_path + 'trl/' + exp_lb['noise'] + '/',
    'avg_pics': p_path +'avg/' + exp_lb['noise'] + '/',
    'trl_pics': p_path +'trl/' + exp_lb['noise'] + '/'
}

NZ_info = {
    'pois': [0],
    'subIDs': ['000'],
    'conditions': NZ_conditions,
    'directories': NZ_paths,
    'k': 5,
    'Rth': 15,
    'tau': 20,
    'avT': 55
}


# Save all of this in a Very Big Dictionary
maind = {
    'path': path,
    'bmasking': BM_info,
    'zbmasking': ZBM_info,
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