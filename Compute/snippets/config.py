import json
import os
import numpy as np
import core as c
import matplotlib.pyplot as plt

from matplotlib import use
from scipy.spatial.distance import squareform

use('kitcat')

print('config')

#L = ['../Cargo/results','avg','BMD','Global','RP','Fixed']
#
#file_path = os.path.join(*L,'recurrence.npz')
#info_path = os.path.join(*L,'info.json')
#
#M = np.load(file_path)
#
#with open(info_path, 'r') as f:
#    info = json.load(f)
#
#def show(file: str, idxs: list):
#
#    m = squareform(M[file][*idxs])
#
#    cut = 0
#    for i in range(0,len(m)):
#        m[i,i] = 1
#        if m[i,0] == 2:
#            cut = i
#            break
#
#    m = m[:cut,:cut]
#
#    plt.imshow(m, cmap = 'Oranges')
#    plt.show()
#
#    del m
#
#    return
