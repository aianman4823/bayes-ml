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
dir_name = 'step3_7/'

# 平均が未知な場合
# 次元数2
mu_truth_d = np.array([25.0, 50.0])
sigma2_dd = np.array([
    [600.0, 100.0],
    [100.0, 400.0]
])
lambda_dd = np.linalg.inv(sigma2_dd)

# グラフ用の点作成
x_0_line = np.linspace(
    mu_truth_d[0] - 3 * np.sqrt(sigma2_dd[0, 0]),
    mu_truth_d[0] + 3 * np.sqrt(sigma2_dd[0, 0]),
    num=500
)
x_1_line = np.linspace(
    mu_truth_d[1] - 3 * np.sqrt(sigma2_dd[1, 1]),
    mu_truth_d[1] + 3 * np.sqrt(sigma2_dd[1, 1]),
    num=500
)

# 格子状のxの値を作成
# x, yの組み合わせを作成できる
x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)

# xの点を作成
x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)
x_dim = x_0_grid.shape
print(x_dim)
print(x_point_arr.shape)
print(x_point_arr[:5])

# 尤度の計算
true_model = st.multivariate_normal.pdf(
    x=x_point_arr, mean=mu_truth_d, cov=np.linalg.inv(lambda_dd)
)

print(x_point_arr.shape)
print(true_model.shape)

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dim,
    true_model,
    xlabel='$x_1$',
    ylabel='$x_2$',
    suptitle='Multivariate Gaussian Distribution',
    title='$\mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' +
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='true_model.png')


# データの生成
N = 50

x_nd = np.random.multivariate_normal(
    mean=mu_truth_d,
    cov=np.linalg.inv(lambda_dd),
    size=N
)

print(x_nd[:5])


U.plot_scatter_2d(x_nd,
                  x_0_grid,
                  x_1_grid,
                  x_dim,
                  model=true_model,
                  suptitle='Multivariate Gaussian Distribution',
                  title='$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' +
                  ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_dd, 5)]) + '$',
                  output_path=base_dir + dir_name,
                  name='scatter_2d.png')


# 事前分布の設定
m_d = np.array([0.0, 0.0])

sigma2_mu_dd = np.array([
    [1000.0, 0.0],
    [0.0, 1000.0]
])

lambda_mu_dd = np.linalg.inv(sigma2_mu_dd)

mu_0_line = np.linspace(
    mu_truth_d[0] - 2 * np.sqrt(sigma2_mu_dd[0, 0]),
    mu_truth_d[0] + 2 * np.sqrt(sigma2_mu_dd[0, 0]),
    num=500
)
mu_1_line = np.linspace(
    mu_truth_d[1] - 2 * np.sqrt(sigma2_mu_dd[1, 1]),
    mu_truth_d[1] + 2 * np.sqrt(sigma2_mu_dd[1, 1]),
    num=500
)

mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)

mu_point_arr = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dims = mu_0_grid.shape
print(mu_dims)

mu_prior = st.multivariate_normal.pdf(
    mu_point_arr,
    mean=m_d,
    cov=np.linalg.inv(lambda_mu_dd)
)

print(mu_point_arr[:10])
print(mu_prior[:10])


U.plot_likelihood_2d(
    mu_0_grid,
    mu_1_grid,
    mu_dims,
    mu_prior,
    xlabel='mu',
    suptitle='Multivariate Gaussian Distribution',
    title='$m=[' + ', '.join([str(m) for m in m_d]) + ']' +
          ', \Lambda_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(lambda_mu_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='prior_mu.png',
    truth=mu_truth_d
)

# 事後分布の計算
lambda_mu_hat_dd = N * lambda_dd + lambda_mu_dd
m_hat_d = np.dot(np.linalg.inv(lambda_mu_hat_dd), (np.dot(lambda_dd, np.sum(x_nd, axis=0)) + np.dot(lambda_mu_dd, m_d)))


print(m_hat_d)
print(lambda_mu_dd)


mu_posterior = st.multivariate_normal.pdf(
    mu_point_arr,
    mean=m_hat_d,
    cov=np.linalg.inv(lambda_mu_hat_dd)
)

print(mu_posterior[:10])

U.plot_likelihood_2d(
    mu_0_grid,
    mu_1_grid,
    mu_dims,
    mu_posterior,
    xlabel='mu1',
    ylabel='mu2',
    suptitle='Multivariate Gaussian Distribution',
    title='$N=' + str(N) +
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' +
          ', \hat{\Lambda}_{\mu}=' + str([list(lmd_d) for lmd_d in np.round(lambda_mu_hat_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='posterior_mu.png',
    truth=mu_truth_d
)

# 予測分布
lambda_star_hat_dd = np.linalg.inv(
    np.linalg.inv(lambda_dd) + np.linalg.inv(lambda_mu_hat_dd)
)
m_star_d = m_hat_d

predict = st.multivariate_normal.pdf(
    x_point_arr,
    mean=m_star_d,
    cov=np.linalg.inv(lambda_star_hat_dd)
)

print(predict[:10])

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dim,
    predict,
    xlabel='x1',
    ylabel='x2',
    output_path=base_dir + dir_name,
    name='predict.png',
    truth=mu_truth_d,
    true_model=true_model,
    x_nd=x_nd
)
