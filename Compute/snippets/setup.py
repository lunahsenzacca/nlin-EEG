import json
import os
import mne
import sys
import numpy as np
import core as c
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
    info_temp.pop('obs_name')
    info_temp.pop('calc_lb')

    pdict(info_temp, title = title)

    return RES, info


if len(sys.argv) > 1:
    exec(open(f'snippets/{sys.argv[1]}.py').read())
