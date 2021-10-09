if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
from bayes.utils import my_makedirs, plot_likelihood
from bayes.models import gauss, gamma_dist
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import bayes.utils as U


# 真のパラメータを設定
mu = 25
lambda_truth = 0.01

x_line = U.make_xline(mu, lambda_truth, n=1000)

true_model = gauss(x_line, mu=mu, sigma=np.sqrt(1 / lambda_truth))

print(x_line[:10])
print(true_model[:10])

U.plot_likelihood(x_line,
                  true_model,
                  suptitle='Gauss Distribution',
                  title='$\mu=' + str(mu) + ', \lambda=' + str(lambda_truth) + '$',
                  output_path='./outputs/step3_5/',
                  name='true.png')

# データの生成
N = 50

# ガウス分布に従うデータをランダム生成
x_n = np.random.normal(loc=mu,
                       scale=np.sqrt(1 / lambda_truth),
                       size=N)
print(x_n[:5])

U.plot_hist(x_n,
            bins=50,
            suptitle='Gauss Distribution',
            title='$N=' + str(N) + ', \mu=' + str(mu) +
            ', \sigma=' + str(np.sqrt(1 / lambda_truth)) + '$',
            output_path='./outputs/step3_5/',
            name='hist.png')

# 事前分布の設定
#  gamma分布を設定
a, b = 1, 1

lambda_line = np.linspace(0, 4 * lambda_truth, num=1000)

prior = gamma_dist(lambda_line, a=a, b=b)


print(lambda_line[:10])
print(prior[:10])

U.plot_likelihood(lambda_line,
                  prior,
                  xlabel='lambda',
                  suptitle='Gamma',
                  title='$a=' + str(a) + ', b=' + str(b) + '$',
                  output_path='./outputs/step3_5/',
                  name='prior.png')


# 事後分布の計算
a_hat = (1 / 2) * N + a
b_hat = (1 / 2) * sum((x_n - mu)**2) + b

posterior = st.gamma.pdf(x=lambda_line, a=a_hat, scale=1 / b_hat)


print(a_hat)
print(b_hat)

print(posterior[:10])
U.plot_likelihood(lambda_line,
                  posterior,
                  suptitle='Gamma Dist',
                  title='$N=' + str(N) + ', a=' + str(a_hat) + ', b=' + str(np.round(b_hat, 1)) + '$',
                  output_path='./outputs/step3_5/',
                  name='posterior.png',
                  truth=lambda_truth
                  )

# 予測分布の計算
mu_s = mu
lambda_s = a_hat / b_hat
v_s = 2 * a_hat

print(mu_s)
print(lambda_s)
print(v_s)

# スチューデントのt分布
predict = st.t.pdf(x=x_line, df=v_s, loc=mu_s, scale=np.sqrt(1 / lambda_s))

U.plot_likelihood(x_line,
                  predict,
                  suptitle="Student's t Distribution",
                  title='$N=' + str(N) + ', \mu_s=' + str(mu_s) +
                  ', \hat{\lambda}_s=' + str(np.round(lambda_s, 3)) +
                  ', \hat{\\nu}_s=' + str(v_s) + '$',
                  output_path='./outputs/step3_5/',
                  name='predict.png',
                  true_model=true_model)
