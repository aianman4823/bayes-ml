if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

from numpy.random.mtrand import gamma
from bayes.utils import my_makedirs, plot_likelihood
from bayes.models import gauss, gamma_dist
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import bayes.utils as U


# 真のパラメータ設定
mu_truth = 25
lambda_truth = 0.01

x_line = U.make_xline(mu_truth, lambda_truth, 1000)

true_model = gauss(x_line, mu=mu_truth, sigma=np.sqrt(1 / lambda_truth))

U.plot_likelihood(x_line,
                  true_model,
                  suptitle='Gauss Dist',
                  title='$\mu=' + str(mu_truth) + ', \lambda=' + str(lambda_truth) + '$',
                  output_path='./outputs/step3_6/',
                  name='truth.png')

# データを生成
N = 50

# ガウス分布に従うデータ生成
x_n = np.random.normal(loc=mu_truth, scale=np.sqrt(1 / lambda_truth), size=N)

print(x_n[:10])

U.plot_hist(x_n,
            bins=50,
            suptitle='Gauss Dist',
            title='$N=' + str(N) + ', \mu=' + str(mu_truth) +
            ', \sigma=' + str(np.sqrt(1 / lambda_truth)) + '$',
            output_path='./outputs/step3_6/',
            name='hist.png')


# 事前分布の設定
# μに対するパラメータの設定
m = 0
beta = 1

# λに対するパラメータの設定
a, b = 1, 1

# λの期待値を計算( gamma分布の期待値の計算)
E_lambda = a / b


mu_line = np.linspace(mu_truth - 50, mu_truth + 50, num=1000)

prior_mu = gauss(mu_line, m, np.sqrt(1 / beta * E_lambda))
print(prior_mu[:10])

U.plot_likelihood(mu_line,
                  prior_mu,
                  xlabel='mu',
                  suptitle='Gauss Dist',
                  title='$m=' + str(m) + ', \\beta=' + str(beta) +
                  ', E[\lambda]=' + str(np.round(E_lambda, 3)) + '$',
                  output_path='./outputs/step3_6/',
                  name='prior_mu.png')

lambda_line = np.linspace(0, 4 * lambda_truth, num=1000)

prior_lambda = gamma_dist(lambda_line, a, b)

U.plot_likelihood(
    lambda_line,
    prior_lambda,
    xlabel='lambda',
    suptitle='Gamma Dist',
    title='$a=' + str(a) + ', b=' + str(b) + '$',
    output_path='./outputs/step3_6/',
    name='prior_lambda.png'
)


# 事後分布の計算
# μ
beta_hat = N + beta
m_hat = (1 / beta_hat) * (np.sum(x_n) + beta * m)

# λ
a_hat = (N / 2) * a
b_hat = (1 / 2) * (np.sum(x_n**2) + beta * m**2 - beta_hat * m_hat**2) + b

# lambdaの期待値の計算
E_lambda_hat = a_hat / b_hat

posterior_mu = gauss(mu_line, m_hat, np.sqrt(1 / (beta_hat * E_lambda_hat)))

U.plot_likelihood(mu_line,
                  posterior_mu,
                  xlabel='mu',
                  suptitle='Gauss Dist',
                  title='$N=' + str(N) + ', \hat{m}=' + str(np.round(m_hat, 1)) +
                  ', \hat{\\beta}=' + str(beta_hat) +
                  ', E[\hat{\lambda}]=' + str(np.round(E_lambda_hat, 3)) + '$',
                  output_path='./outputs/step3_6/',
                  name='posterior_mu.png',
                  truth=mu_truth)

posterior_lambda = st.gamma.pdf(x=lambda_line,
                                a=a_hat,
                                scale=1 / b_hat)

U.plot_likelihood(lambda_line,
                  posterior_lambda,
                  xlabel='lambda',
                  suptitle='Gamma Dist',
                  title='$N=' + str(N) + ', a=' + str(a_hat) + ', b=' + str(np.round(b_hat, 1)) + '$',
                  output_path='./outputs/step3_6/',
                  name='posterior_lambda.png',
                  truth=lambda_truth)

# 予測分布の計算
# パラメータ
m_s_hat = m_hat
lambda_s_hat = (beta_hat * a_hat) / ((1 + beta_hat) * b_hat)
v_s_hat = 2 * a_hat

predict = st.t.pdf(x=x_line, df=v_s_hat, loc=m_s_hat, scale=np.sqrt(1 / lambda_s_hat))

U.plot_likelihood(x_line,
                  predict,
                  suptitle="Sturdent's t dist",
                  title='$N=' + str(N) + ', \hat{\mu}_s=' + str(np.round(m_s_hat, 1)) +
                  ', \hat{\lambda}_s=' + str(np.round(lambda_s_hat, 3)) +
                  ', \hat{\\nu}_s=' + str(v_s_hat) + '$',
                  output_path='./outputs/step3_6/',
                  name='predict.png',
                  true_model=true_model)
