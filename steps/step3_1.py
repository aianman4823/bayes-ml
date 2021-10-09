if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import bayes.models as M
import numpy as np
from scipy import integrate
import scipy.stats as stats
import matplotlib.pyplot as plt
import bayes.utils
import math

# 真のパラメータ
mu_truth = 0.25
x_point = np.array([0, 1])

# ベルヌーイ(尤度関数として考えている) -> つまり、muは共役事前分布ベータに従う
true_model = np.array([1 - mu_truth, mu_truth])
print(true_model)

fig_path = 'outputs/images/step3_1'
bayes.utils.my_makedirs(fig_path)

fig = plt.figure(figsize=(12, 9))
plt.bar(x=x_point, height=true_model)
plt.xlabel('x')
plt.ylabel('prob')
plt.xticks(ticks=x_point, labels=x_point)
plt.suptitle('Bernoulli Distribution', fontsize=20)
plt.title('$\mu=' + str(mu_truth) + '$', loc='left')
plt.ylim(0.0, 1.0)
plt.savefig(fig_path + '/bernoulli_likelihood.png')


# データ生成
N = 50
# ベルヌーイ分布に従うデータ
x_n = np.random.binomial(n=1, p=mu_truth, size=N)
# 事前分布の設定
# 尤度関数(ベルヌーイ)に対する共役事前分布を設定
a = 1.0
b = 1.0
mu_line = np.arange(0.0, 1.001, 0.001)
ln_C_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
beta = np.exp(ln_C_beta) * mu_line ** (a - 1) * (1 - mu_line) ** (b - 1)
print(beta)
# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 事前分布を作図
plt.plot(mu_line, beta, color='purple')
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Beta Distribution', fontsize=20)
plt.title('a=' + str(a) + ', b=' + str(b), loc='left')
plt.savefig(fig_path + '/beta_prior.png')


# 事後分布の計算
a_hat = np.sum(x_n) + a
b_hat = N - np.sum(x_n) + b
print(a_hat, b_hat)

ln_C_beta = math.lgamma(a_hat + b_hat) - math.lgamma(a_hat) - math.lgamma(b_hat)
posterior_beta = np.exp(ln_C_beta) * mu_line ** (a_hat - 1) * (1 - mu_line) ** (b_hat - 1)
print(posterior_beta)

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 事後分布を作図
plt.plot(mu_line, posterior_beta, color='purple')  # 事後分布
plt.vlines(x=mu_truth, ymin=0, ymax=max(posterior_beta), linestyles='--', color='red')  # 真のパラメータ
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Beta Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(b_hat) + '$', loc='left')
plt.savefig(fig_path + '/posterior.png')


mu_pred = (a_hat) / (a_hat + b_hat)
print(mu_pred)
pred_x = np.array([1 - mu_pred, mu_pred])
print(pred_x)
# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 予測分布を作図
plt.bar(x=x_point, height=true_model, label='true', alpha=0.5,
        color='white', edgecolor='red', linestyle='dashed')  # 真のモデル
plt.bar(x=x_point, height=pred_x, label='predict', alpha=0.5,
        color='purple')  # 予測分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Bernoulli Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \hat{\mu}_{*}=' + str(np.round(mu_pred, 2)) + '$', loc='left')
plt.ylim(0.0, 1.0)
plt.legend()
plt.savefig(fig_path + '/pred.png')
