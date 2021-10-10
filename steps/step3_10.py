if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from os import name, terminal_size
from numpy.core.fromnumeric import mean

from scipy.stats.stats import power_divergence

from bayes.utils import make_xline, my_makedirs, plot_likelihood
from bayes.models import gauss, gamma_dist
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import bayes.utils as U

base_dir = './outputs/'
dir_name = 'step3_10/'

print(U.x_vector(np.arange(1, 5), 4))

# 真のパラメータ設定
# 真のパラメータの次元数
M_truth = 4

# パラメータの初期値
w_truth_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
)
print(w_truth_m)

x_line = np.arange(-3.0, 3.0, step=0.01)
x_truth_arr = U.x_vector(x_line, M_truth)
print(x_truth_arr[:5])
print(w_truth_m.shape)
print(x_truth_arr.shape)


y_line = np.dot(w_truth_m.reshape((1, M_truth)), x_truth_arr.T).flatten()

print(np.round(y_line[:5], 2))

U.plot_linear(
    x_line,
    y_line,
    suptitle='Observation Model',
    title='w=[' + ', '.join([str(w) for w in np.round(w_truth_m, 2).flatten()]) + ']',
    xlabel='x',
    ylabel='y',
    output_path=base_dir + dir_name,
    name='true_linear_model.png'
)
