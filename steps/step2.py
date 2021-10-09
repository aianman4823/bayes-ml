if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import bayes.models as M
import numpy as np
from scipy import integrate
import scipy.stats as stats


def f(x, mu=0.5):
    y = x ** 2 + 1
    # ベルヌーイ分布
    y *= (mu ** x) * (1 - mu)**(1 - x)
    return y


# 期待値の定義
mu = 0.5
lower = -1
upper = 1
Ex, err = integrate.quad(f, lower, upper, args=(mu))
print(Ex)


# 基本的な期待値と分散

def f1(x, mu=0.5):
    y = x
    y *= (mu ** x) * (1 - mu)**(1 - x)
    return y


mu = 0.1
Ex, err = integrate.quad(f1, lower, upper, args=(mu))
print(Ex)


def f2(x, mu=0.5):
    y = np.array(x) * np.array(x).T
    y *= (mu ** x) * (1 - mu)**(1 - x)
    return y


mu = 0.1
Ex, err = integrate.quad(f2, lower, upper, args=(mu))
print(Ex)


def f3(x, mu=0.5):
    y = np.array(x).T
    y *= (mu ** x) * (1 - mu)**(1 - x)
    return y


mu = 0.1
Ex, err = integrate.quad(f3, lower, upper, args=(mu))
print(Ex)


# エントロピー
def entropy(x, mu=0.5):
    y = (mu ** x) * (1 - mu)**(1 - x)
    y *= np.log((mu ** x) * (1 - mu)**(1 - x))
    y *= -1
    return y


mu = 0.1
Ex, err = integrate.quad(entropy, lower, upper, args=(mu))
print(Ex)


def f4(x, p=(1 / 3)):
    c1 = np.count_nonzero(x == 1)
    c0 = x.size - c1
    return c0, c1, p, (1 - p)


def cal_f4(x):
    c0, c1, p, p_1 = f4(x)
    return -np.sum(c0 * p * np.log(p) + c1 * p_1 * np.log(p_1))


x = np.array([0, 1])
ans = cal_f4(x)
print(ans)


# サンプリングによる期待値の近似
def cal_f4_sampling(x):
    c0, c1, p, p_1 = f4(x)
    return -np.sum(c1 * np.log(p) + c0 * np.log(p_1))


x = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
sum_ans = cal_f4_sampling(x)
print(sum_ans / len(x))
