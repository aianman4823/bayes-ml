if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from os import name

from bayes.utils import make_xline, my_makedirs, plot_likelihood
from bayes.models import gauss, gamma_dist
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import bayes.utils as U

base_dir = './outputs/'
dir_name = 'step3_8/'

# 分散共分散行列が未知の場合

