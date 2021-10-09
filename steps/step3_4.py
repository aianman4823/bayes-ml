if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bayes.utils import my_makedirs
import numpy as np
from scipy.stats import norm  # 1次元ガウス分布
import matplotlib.pyplot as plt
import bayes.utils as U

# 真のパラメータ
mu_truth = 25
# 制度パラメータ
lmd = 0.01

# 作図用のxの値を設定
x_line = np.linspace(mu_truth - 4 * np.sqrt(1 / lmd),
                     mu_truth + 4 * np.sqrt(1 / lmd),
                     num=1000)

# 尤度を計算
# 正規化項
ln_C_N = -(1 / 2) * (np.log(1) - np.log(lmd) + np.log(2 * np.pi))
true_model = np.exp(ln_C_N - (1 / 2) * ((x_line - mu_truth)**2 / (1 / lmd)))


# 尤度を作図
U.plot_likelihood(x_line,
                  true_model,
                  suptitle='Gaussian Distribution',
                  title='$\mu=' + str(mu_truth) + ', \lambda=' + str(lmd) + '$',
                  output_path='./outputs/step3_4',
                  name='gauss.png')


# 観測データ生成
N = 50

# ガウス分布に従うデータを生成
x_n = np.random.normal(loc=mu_truth, scale=np.sqrt(1 / lmd), size=N)

U.plot_hist(x_n,
            bins=50,
            xlabel='x',
            ylabel='count',
            suptitle='Gaussian Distribution',
            title='$N=' + str(N) + ', \mu=' + str(mu_truth) +
            ', \sigma=' + str(np.sqrt(1 / lmd)) + '$',
            output_path='./outputs/step3_4/',
            name='hist_gauss.png')

# 事前分布の設定
# 尤度に対する共役事前分布
m = 0
lambda_mu = 0.001

mu_line = np.linspace(mu_truth - 50,
                      mu_truth + 50,
                      num=1000)

ln_C_N = -0.5 * (np.log(2 * np.pi) - np.log(lambda_mu))
prior = np.exp(ln_C_N - 0.5 * lambda_mu * (mu_line - m) ** 2)

print(mu_line[:10])
print(prior[:10])

U.plot_likelihood(mu_line,
                  prior,
                  suptitle='Gaussian Distribution',
                  title='$m=' + str(m) + ', \lambda_{\mu}=' + str(lambda_mu) + '$',
                  output_path='./outputs/step3_4/',
                  name='prior.png')


# 事後分布の計算
lambda_mu_hat = N * lmd + lambda_mu
m_hat = (lmd * sum(x_n) + lambda_mu * m) / lambda_mu_hat

print(lambda_mu_hat)
print(m_hat)

# 求めた事後分布を計算
ln_C_N = -0.5 * (np.log(2 * np.pi) - np.log(lambda_mu_hat))
posterior = np.exp(ln_C_N - 0.5 * lambda_mu_hat * (mu_line - m_hat)**2)

print(posterior[:10])

U.plot_likelihood(mu_line,
                  posterior,
                  suptitle='Gaussian Distribution',
                  title='$\hat{m}=' + str(np.round(m_hat, 1)) +
                  ', \hat{\lambda}_{\mu}=' + str(np.round(lambda_mu_hat, 3)) + '$',
                  output_path='./outputs/step3_4/',
                  name='posterior.png',
                  truth=mu_truth)

# 予測分布の計算
lambda_star_hat = lmd * lambda_mu_hat / (lmd + lambda_mu_hat)
mu_star_hat = m_hat


print(lambda_star_hat)
print(mu_star_hat)

ln_C_N = -0.5 * (np.log(2 * np.pi) - np.log(lambda_star_hat))
predict = np.exp(ln_C_N - 0.5 * lambda_star_hat * (x_line - mu_star_hat)**2)

print(predict[:10])

U.plot_likelihood(x_line,
                  predict,
                  suptitle='Gaussian Distribution',
                  title='$\hat{\mu}_{*}=' + str(np.round(mu_star_hat, 1)) +
                  ', \hat{\lambda}_{*}=' + str(np.round(lambda_star_hat, 3)) + '$',
                  output_path='./outputs/step3_4/',
                  name='predict.png',
                  true_model=true_model)
