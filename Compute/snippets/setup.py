import json
import os
import mne
import sys
import numpy as np
import core as c
import matplotlib.pyplot as plt

from matplotlib import use

use('kitcat')

if len(sys.argv) > 1:
    exec(open(f'snippets/{sys.argv[1]}.py').read())
