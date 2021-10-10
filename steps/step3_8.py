if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from os import name

from scipy.stats.stats import power_divergence

from bayes.utils import make_xline, my_makedirs, plot_likelihood
from bayes.models import gauss, gamma_dist
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import bayes.utils as U

base_dir = './outputs/'
dir_name = 'step3_8/'

# 分散共分散行列が未知の場合

mu_d = np.array([25.0, 50.0])

# 分散共分散行列
sigma2_truth_dd = np.array([
    [600.0, 100.0],
    [100.0, 400.0],
])

# 精度行列
lambda_truth_dd = np.linalg.inv(sigma2_truth_dd)

x_0_line = np.linspace(
    mu_d[0] - 3 * np.sqrt(sigma2_truth_dd[0, 0]),
    mu_d[0] + 3 * np.sqrt(sigma2_truth_dd[0, 0]),
    num=500
)

x_1_line = np.linspace(
    mu_d[1] - 3 * np.sqrt(sigma2_truth_dd[1, 1]),
    mu_d[1] + 3 * np.sqrt(sigma2_truth_dd[1, 1]),
    num=500
)

x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)

x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)

x_dims = x_0_grid.shape
print(x_dims)

print(x_point_arr[:5])

true_model = st.multivariate_normal.pdf(
    x_point_arr,
    mean=mu_d,
    cov=np.linalg.inv(lambda_truth_dd)
)

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    true_model,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss dist',
    title='$\mu=[' + ', '.join([str(mu) for mu in mu_d]) + ']' +
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='truth.png',
)


# データの生成
N = 50

x_nd = np.random.multivariate_normal(mu_d,
                                     np.linalg.inv(lambda_truth_dd),
                                     size=N)

print(x_nd.shape)

U.plot_scatter_2d(
    x_nd,
    x_0_grid,
    x_1_grid,
    x_dims,
    true_model,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_d]) + ']' +
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='scatter_2d.png'
)


# 事前分布の設定
nu = 2.0
w_dd = np.array([
    [0.0005, 0],
    [0, 0.0005]
])

E_lambda_dd = nu * w_dd

prior = st.multivariate_normal.pdf(
    x=x_point_arr,
    mean=mu_d,
    cov=np.linalg.inv(E_lambda_dd)
)
# prior = st.wishart.pdf(
#     x_nd,
#     df=nu,
#     scale=w_dd
# )

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    prior,
    xlabel='x1',
    ylabel='x2',
    suptitle='Wishart',
    title='$\\nu=' + str(nu) +
          ', W=' + str([list(w_d) for w_d in np.round(w_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='prior.png',
    true_model=true_model
)

# 事後分布の計算
w_hat_dd = np.linalg.inv(
    np.dot((x_nd - mu_d).T, (x_nd - mu_d)) + np.linalg.inv(w_dd)
)
nu_hat = N + nu

E_lambda_hat_dd = nu_hat * w_hat_dd

posterior = st.multivariate_normal.pdf(
    x_point_arr,
    mean=mu_d,
    cov=np.linalg.inv(E_lambda_hat_dd)
)

print(posterior)

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    posterior,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$\\nu=' + str(nu_hat) +
          ', W=' + str([list(w_hat) for w_hat in np.round(w_hat_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='posterior.png',
    true_model=true_model
)

# 予測分布
# 次元を取得
D = len(mu_d)

mu_s_d = mu_d
lambda_s_hat_dd = (1 - D + nu_hat) * w_hat_dd
nu_s_hat = 1 - D + nu_hat

predict = st.multivariate_t.pdf(
    x=x_point_arr,
    loc=mu_s_d,
    shape=np.linalg.inv(lambda_s_hat_dd),
    df=nu_s_hat
)

print(predict)

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    predict,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$\\nu=' + str(nu_hat) +
          ', W=' + str([list(w_hat) for w_hat in np.round(w_hat_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='predict.png',
    true_model=true_model,
    x_nd=x_nd
)
