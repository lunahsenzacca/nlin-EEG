import json
import os
import mne
import sys
import numpy as np
import core as c
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from rich import print as pp

from style import pdict

from init import maind

from matplotlib import use

import inquirer as inq

use('kitcat')

def show_info(info: dict, keys: list):

    print('\nHere\'s some info\n')

    pp({key: info[key] for key in keys})

    return

def load(obs_name: str, path: str | None = None):

    if path is None:

        saved = []
        for root, dirs, files in os.walk('../Cargo/results', topdown = False):
            for name in files:
                if name == 'info.json':
                    relpath = os.path.relpath(root, start = '../Cargo/results')
                    if f'/{maind['obs_lb'][obs_name]}' in relpath:
                        saved.append(relpath)

        if len(saved) == 0:
            print('Found no results')

            return None, None

        input = [
            inq.List('saved_opt',
                     message = 'Found the following saved results',
                     choices = saved
            )
        ]

        print('')

        saved_opt = inq.prompt(input)['saved_opt']
        res_path = '../Cargo/results/' + saved_opt + '/'

        res_file = res_path + f'{obs_name}.npz'
        info_file = res_path + 'info.json'

    else:

        res_file = os.path.join('../Cargo/results',path,f'{obs_name}.npz')
        info_file = os.path.join('../Cargo/results',path,'info.json')

    RES = np.load(res_file)

    with open(info_file, 'r') as f:
        info = json.load(f)

    title = maind['obs_nm'][info['obs_name']]
    if info['calc_lb'] is not None:
        title = title + f'\n({info['calc_lb']})'

    info_temp = info.copy()

    pop = ['obs_name','calc_lb','sub_list','t','clst_lb','exp_lb','avg_trials']

    for key in list(info_temp.keys()):

        if key in pop:
            info_temp.pop(key)

    pdict(info_temp, title = title)

    return RES, info

def group(RES, info):

    M = []

    for i in range(0,len(info['sub_list'])):
        M.append([RES[f'arr_{2*i}'],RES[f'arr_{2*i + 1}']])

    return np.asarray(M)

def get_mne_info(exp_lb):

    ex_file = os.listdir(f'../Cargo/toMNE/avg/{exp_lb}')[0]
    _ = mne.read_evokeds(f'../Cargo/toMNE/avg/{exp_lb}/{ex_file}', verbose = False)[0]

    return _.info

if len(sys.argv) > 1:
    exec(open(f'snippets/{sys.argv[1]}.py').read())
