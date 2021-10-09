if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from numpy.core.fromnumeric import size
from scipy.stats.stats import pointbiserialr
from bayes.utils import my_makedirs
from os import W_OK
import numpy as np
import math
from scipy.stats import dirichlet, multinomial
import matplotlib.pyplot as plt


# カテゴリ分布を尤度関数にする
# 次元を固定
K = 3
# 真のパラメータ指定
pi_truth_k = np.array([0.3, 0.5, 0.2])

k_line = np.arange(1, K + 1)

# 1 of K representation(1ofK表現を作成)
s_kk = np.identity(K)
# np.prodは要素の積をとる
print(np.prod(pi_truth_k ** s_kk, axis=1))
print(multinomial.pmf(x=s_kk, n=1, p=pi_truth_k))


fig_path = 'outputs/images/step3_2'
my_makedirs(fig_path)
fig = plt.figure(figsize=(12, 9))
plt.bar(x=k_line, height=pi_truth_k, color='purple')  # 真のモデル
plt.xlabel('k')
plt.ylabel('prob')
plt.xticks(ticks=k_line, labels=k_line)  # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$\pi=(' + ', '.join([str(k) for k in pi_truth_k]) + ')$', loc='left')
plt.ylim(0, 1)
plt.savefig(fig_path + '/category_likelihood.png')

# データ生成
N = 50
s_nk = np.random.multinomial(n=1, pvals=pi_truth_k, size=N)
print(s_nk[:5])
print(np.sum(s_nk, axis=0))


# 事前分布の設定
# 尤度関数をカテゴリ分布としたので、共役事前分布はディクレリ分布となる
# >0であればなんでも(ハイパラメータ)
alpha_k = np.array([1.0, 1.0, 1.0])
point_vec = np.arange(0.0, 1.001, 0.02)

X, Y, Z = np.meshgrid(point_vec, point_vec, point_vec)
# 確率密度の計算用にまとめる
pi_point = np.array([list(X.flatten()), list(Y.flatten()), list(Z.flatten())]).T
pi_point = pi_point[1:, :]
pi_point /= np.sum(pi_point, axis=1, keepdims=True)
pi_point = np.unique(pi_point, axis=0)

# 事前分布の確率蜜度計算
ln_C_dir = math.lgamma(np.sum(alpha_k)) - np.sum([math.lgamma(a) for a in alpha_k])
prior = np.exp(ln_C_dir) * np.prod(pi_point ** (alpha_k - 1), axis=1)

# prior = np.array([
#     dirichlet.pdf(x=pi_point[i], alpha=alpha_k) for i in range(len(pi_point))
# ])
print(prior)

# 三角座標に変換
tri_x = pi_point[:, 1] + pi_point[:, 2] / 2
tri_y = np.sqrt(3) * pi_point[:, 2] / 2

fig = plt.figure(figsize=(12, 9))
plt.scatter(tri_x, tri_y, c=prior, cmap='jet')  # 事前分布
plt.xlabel('$\pi_1, \pi_2$')  # x軸ラベル
plt.ylabel('$\pi_1, \pi_3$')  # y軸ラベル
plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)'])  # x軸目盛
plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)'])  # y軸目盛
plt.suptitle('Dirichlet Distribution', fontsize=20)
plt.title('$\\alpha=(' + ', '.join([str(k) for k in alpha_k]) + ')$', loc='left')
plt.colorbar()  # 凡例
plt.gca().set_aspect('equal')  # アスペクト比
plt.savefig(fig_path + '/dirichlet_dist.png')


# 事後分布の計算
alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
print(alpha_hat_k)

# 事後分布の確率蜜度計算
ln_C_dir = math.lgamma(np.sum(alpha_hat_k)) - np.sum([math.lgamma(a) for a in alpha_hat_k])
posterior = np.exp(ln_C_dir) * np.prod(pi_point ** (alpha_hat_k - 1), axis=1)

# posterior = np.array([
#     dirichlet.pdf(x=pi_point[i], alpha=alpha_hat_k) for i in range(len(pi_point))
# ])
print(posterior)

# 真のパラメータの値を三角座標に変換
tri_x_truth = pi_truth_k[1] + pi_truth_k[2] / 2
tri_y_truth = np.sqrt(3) * pi_truth_k[2] / 2

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 事後分布を作図
plt.scatter(tri_x, tri_y, c=posterior, cmap='jet')  # 事後分布
plt.xlabel('$\pi_1, \pi_2$')  # x軸ラベル
plt.ylabel('$\pi_1, \pi_3$')  # y軸ラベル
plt.xticks(ticks=[0.0, 1.0], labels=['(1, 0, 0)', '(0, 1, 0)'])  # x軸目盛
plt.yticks(ticks=[0.0, 0.87], labels=['(1, 0, 0)', '(0, 0, 1)'])  # y軸目盛
plt.suptitle('Dirichlet Distribution', fontsize=20)
plt.title('$\\alpha=(' + ', '.join([str(k) for k in alpha_hat_k]) + ')$', loc='left')
plt.colorbar()  # 凡例
plt.gca().set_aspect('equal')  # アスペクト比
plt.scatter(tri_x_truth, tri_y_truth, marker='x', color='black', s=200)  # 真のパラメータ
plt.savefig(fig_path + '/posterior.png')


# 予測分布の計算
pi_hat_k = alpha_hat_k / np.sum(alpha_hat_k)
pi_hat_k = (np.sum(s_nk, axis=0) + alpha_k) / np.sum(np.sum(s_nk, axis=0) + alpha_k)
print(pi_hat_k)


fig = plt.figure(figsize=(12, 9))
plt.bar(x=k_line, height=pi_truth_k, label='truth',
        alpha=0.5, color='white', edgecolor='red', linestyle='dashed')  # 真のモデル
plt.bar(x=k_line, height=pi_hat_k, label='predict',
        alpha=0.5, color='purple')  # 予測分布
plt.xlabel('k')
plt.ylabel('prob')
plt.xticks(ticks=k_line, labels=k_line)  # x軸目盛
plt.suptitle('Categorical Distribution', fontsize=20)
plt.title('$N=' + str(N) +
          ', \hat{\pi}_{*}=(' + ', '.join([str(k) for k in np.round(pi_hat_k, 2)]) + ')$',
          loc='left')
plt.ylim(0, 1)
plt.savefig(fig_path + '/pred.png')
