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
dir_name = 'step3_9/'


# truth
mu_truth_d = np.array([25.0, 50.0])

sigma2_truth_dd = np.array([
    [600.0, 100.0],
    [100.0, 400.0]
])

lambda_truth_dd = np.linalg.inv(sigma2_truth_dd)

x_0_line = np.linspace(
    mu_truth_d[0] - 3 * np.sqrt(sigma2_truth_dd[0, 0]),
    mu_truth_d[0] + 3 * np.sqrt(sigma2_truth_dd[0, 0]),
    num=500
)
x_1_line = np.linspace(
    mu_truth_d[1] - 3 * np.sqrt(sigma2_truth_dd[1, 1]),
    mu_truth_d[1] + 3 * np.sqrt(sigma2_truth_dd[1, 1]),
    num=500
)

x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)
x_dims = x_0_grid.shape

x_point_array = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)

true_model = st.multivariate_normal.pdf(
    x=x_point_array,
    mean=mu_truth_d,
    cov=np.linalg.inv(lambda_truth_dd)
)
U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    true_model,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$\mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' +
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='truth.png'
)

# データ生成
N = 50

x_nd = np.random.multivariate_normal(
    mean=mu_truth_d,
    cov=np.linalg.inv(lambda_truth_dd),
    size=N
)

print(x_nd.shape)
print(x_nd[:10])

U.plot_scatter_2d(
    x_nd,
    x_0_grid,
    x_1_grid,
    x_dims,
    true_model,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$N=' + str(N) + ', \mu=[' + ', '.join([str(mu) for mu in mu_truth_d]) + ']' +
          ', \Lambda=' + str([list(lmd_d) for lmd_d in np.round(lambda_truth_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='scatter_2d.png',
)

# 事前分布
mu_d = np.array([0.0, 0.0])
beta = 1.0

# lambdaの事前分布のパラメータ設定
nu = 2.0
w_dd = np.array([
    [0.0005, 0],
    [0, 0.0005]
])

E_lambda_dd = nu * w_dd

E_sigma2_mu_dd = np.linalg.inv(beta * E_lambda_dd)

mu_0_line = np.linspace(
    mu_truth_d[0] - 2 * np.sqrt(E_sigma2_mu_dd[0, 0]),
    mu_truth_d[0] + 2 * np.sqrt(E_sigma2_mu_dd[0, 0]),
    num=500
)
mu_1_line = np.linspace(
    mu_truth_d[1] - 2 * np.sqrt(E_sigma2_mu_dd[1, 1]),
    mu_truth_d[1] + 2 * np.sqrt(E_sigma2_mu_dd[1, 1]),
    num=500
)
mu_0_grid, mu_1_grid = np.meshgrid(mu_0_line, mu_1_line)
mu_point_array = np.stack([mu_0_grid.flatten(), mu_1_grid.flatten()], axis=1)
mu_dims = mu_0_grid.shape
print(mu_dims)

prior_mu = st.multivariate_normal.pdf(
    mu_point_array,
    mean=mu_d,
    cov=np.linalg.inv(E_lambda_dd)
)

print(mu_point_array)
print(prior_mu)

U.plot_likelihood_2d(
    mu_0_grid,
    mu_1_grid,
    mu_dims,
    prior_mu,
    xlabel='mu1',
    ylabel='mu2',
    suptitle='Multi Gauss Dist',
    title='$\\beta=' + str(beta) +
          ', m=[' + ', '.join([str(m) for m in mu_d]) + ']' +
          ', E[\Lambda]=' + str([list(lmd_d) for lmd_d in np.round(E_lambda_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='prior_mu.png',
    truth=mu_truth_d
)

# muの期待値を計算
E_mu_d = mu_d

prior_lambda = st.multivariate_normal.pdf(
    x_point_array,
    E_mu_d,
    np.linalg.inv(beta * E_lambda_dd)
)

print(prior_lambda)

U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    prior_lambda,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$m=[' + ', '.join([str(m) for m in mu_d]) + ']' +
          ', \\nu=' + str(nu) +
          ', W=' + str([list(w_d) for w_d in np.round(w_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='prior_lambda.png',
    true_model=true_model,
    truth=mu_truth_d
)


# 事後分布の計算
# muの事後分布のパラメータ設定
beta_hat = N + beta
m_hat_d = (1 / beta_hat) * (np.sum(x_nd, axis=0) + beta * mu_d)

print(beta_hat)
print(m_hat_d)

w_hat_dd = np.linalg.inv(
    np.dot(x_nd.T, x_nd)
    + beta * np.dot(mu_d.reshape([2, 1]), mu_d.reshape([2, 1]).T)
    - beta_hat * np.dot(m_hat_d.reshape([2, 1]), m_hat_d.reshape([2, 1]).T)
    + np.linalg.inv(w_dd)
)

nu_hat = N + nu

print(w_hat_dd)
print(nu_hat)

# muの事後分布
E_lambda_hat_dd = nu_hat * w_hat_dd
posterior_mu = st.multivariate_normal.pdf(
    x=mu_point_array,
    mean=m_hat_d,
    cov=np.linalg.inv(beta_hat * E_lambda_hat_dd)
)

U.plot_likelihood_2d(
    mu_0_grid,
    mu_1_grid,
    mu_dims,
    posterior_mu,
    xlabel='mu1',
    ylabel='mu2',
    suptitle='Multi Gauss Dist',
    title='$N=' + str(N) +
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' +
          ', \hat{\\beta}=' + str(beta_hat) +
          ', E[\hat{\Lambda}]=' + str([list(lmd_d) for lmd_d in np.round(E_lambda_hat_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='posterior_mu.png',
    truth=mu_truth_d
)


# lambdaの事後分布
E_mu_hat_d = m_hat_d
print(E_mu_hat_d)


posterior_lambda = st.multivariate_normal.pdf(
    x_point_array,
    mean=E_mu_hat_d,
    cov=np.linalg.inv(E_lambda_hat_dd)
)


U.plot_likelihood_2d(
    x_0_grid,
    x_1_grid,
    x_dims,
    posterior_lambda,
    xlabel='x1',
    ylabel='x2',
    suptitle='Multi Gauss Dist',
    title='$N=' + str(N) +
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_d, 1)]) + ']' +
          ', \hat{\\nu}=' + str(nu_hat) +
          ', \hat{W}=' + str([list(w_d) for w_d in np.round(w_hat_dd, 5)]) + '$',
    output_path=base_dir + dir_name,
    name='posterior_lambda.png',
    true_model=true_model,
    truth=mu_truth_d
)


# 予測分布の計算
D = len(mu_d)

mu_s_d = m_hat_d
lambda_s_hat_dd = (1 - D + nu_hat) * beta_hat / (1 + beta_hat) * w_hat_dd
nu_s_hat = 1 - D + nu_hat


print(mu_s_d)
print(lambda_s_hat_dd)
print(nu_s_hat)


predict = st.multivariate_t.pdf(
    x=x_point_array,
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
    suptitle="Multi sturdent's t Dist",
    title='$N=' + str(N) +
          ', \hat{\mu}_s=[' + ', '.join([str(mu) for mu in np.round(mu_s_d, 1)]) + ']' +
          ', \hat{\Lambda}_s=' + str([list(lmd_d) for lmd_d in np.round(lambda_s_hat_dd, 5)]) +
          ', \hat{\\nu}_s=' + str(nu_s_hat) + '$',
    output_path=base_dir + dir_name,
    name='predict.png',
    truth=mu_truth_d,
    true_model=true_model,
    x_nd=x_nd
)
