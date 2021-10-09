if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bayes.utils import my_makedirs
import numpy as np
import math
from scipy.stats import poisson, gamma, nbinom
import matplotlib.pyplot as plt


# 尤度関数にポアソン分布を仮定
lambda_truth = 4.0
x_line = np.arange(4 * lambda_truth)

# true_model = np.exp(x_line * np.log(lambda_truth) - [math.lgamma(x + 1) for x in x_line] - lambda_truth)
true_model = poisson.pmf(k=x_line, mu=lambda_truth)
print(np.round(true_model, 3))

fig_path = 'outputs/images/step3_3'
fig = plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=true_model, color='purple')  # 尤度
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Distribution', fontsize=20)
plt.title('$\lambda=' + str(lambda_truth) + '$', loc='left')
if not os.path.exists(fig_path):
    os.mkdir(fig_path)
plt.savefig(fig_path + '/poi_likelihood.png')

# データの生成
N = 50
x_n = np.random.poisson(lam=lambda_truth, size=N)
print(x_n[:10])


fig = plt.figure(figsize=(12, 9))
plt.bar(x_line, [np.sum(x_n == x) for x in x_line])  # 観測データ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.savefig(fig_path + '/hist.png')


# 共役事前分布を設定(ガンマ分布)
a, b = 1, 1
lambda_line = np.arange(0, 2 * lambda_truth, 0.001)

ln_C_gam = a * np.log(b) - math.lgamma(a)
prior = np.exp(
    (a - 1) * np.log(lambda_line) - b * lambda_line + ln_C_gam
)
# prior = gamma.pdf(x=lambda_line, a=a, scale=1 / b)
# log(0)がnanになる
print(np.round(prior, 3))

fig = plt.figure(figsize=(12, 9))
plt.plot(lambda_line, prior, color='purple')  # 事前分布
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + "$", loc='left')
plt.savefig(fig_path + '/prior_gamma.png')


# 事後分布の計算
a_hat = np.sum(x_n) + a
b_hat = N + b
print(a_hat, b_hat)

ln_C_gam = a_hat * np.log(b_hat) - math.lgamma(a_hat)
posterior = np.exp(
    (a_hat - 1) * np.log(lambda_line) - b_hat * lambda_line + ln_C_gam
)
# posterior = gamma.pdf(x=lambda_line, a=a_hat, scale=1 / b_hat)
print(np.round(posterior, 5))

fig = plt.figure(figsize=(12, 9))
plt.plot(lambda_line, posterior, color='purple')  # 事後分布
plt.vlines(x=lambda_truth, ymin=0, ymax=max(posterior), color='red', linestyle='--')  # 真のパラメータ
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(b_hat) + "$", loc='left')
plt.savefig(fig_path + '/posterior_lambda.png')

# 予測分布(負の二項分布)
r_hat = a_hat
p_hat = 1 / (b_hat + 1)

print(r_hat, p_hat)

ln_C_NB = np.array([math.lgamma(x + r_hat) - math.lgamma(x + 1)for x in x_line]) - math.lgamma(r_hat)
predict = np.exp(
    ln_C_NB + r_hat * np.log(1 - p_hat) + x_line * np.log(p_hat)
)
# predict = nbinom.pmf(k=x_line, n=r_hat, p=1 - p_hat)
print(np.round(predict, 3))

fig = plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=true_model, label='true', alpha=0.5, color='white',
        edgecolor='red', linestyle='--')  # 真の分布
plt.bar(x=x_line, height=predict, label='predict', alpha=0.5, color='purple')  # 予測分布
plt.xlabel('x')
plt.ylabel('prod')
plt.suptitle('Negative Binomial Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{r}=' + str(r_hat) + ', \hat{p}=' + str(np.round(p_hat, 3)) + '$', loc='left')
plt.savefig(fig_path + '/pred.png')
