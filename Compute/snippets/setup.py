import os
import sys
import numpy as np
import core as c
import matplotlib.pyplot as plt

from matplotlib import use

use('kitcat')

exec(open(f'snippets/{sys.argv[1]}.py').read())
