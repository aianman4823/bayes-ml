import numpy as np
from itertools import permutations
import math


def gauss(x, mu=0, sigma=1):
    c = 1.0 / np.sqrt(2 * np.pi) / sigma
    idx = - 0.5 * ((x - mu) / sigma) ** 2
    return np.exp(idx) * c


def bernouulli(x, mu=0.5):
    return (mu ** x) * (1 - mu)**(1 - x)


def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


def bin(x, M=10, m=5, mu=0.5):
    y = combinations_count(M, m)
    y *= mu ** m
    y *= (1 - mu) ** (M - m)
    return y


# ポアソン分布
def poi(x, lumda=0.1):
    y = lumda**x
    y /= math.factorial(x)
    y *= np.exp(-lumda)
    return y


def log_poi(x, lumda=0.1):
    y = x * np.log(lumda)
    y -= np.log(math.factorial(x))
    y -= lumda
    return y

# ガンマ関数


def gammma(x):
    y = math.gamma(x)
    return y
# ベータ分布


def beta(mu, a, b):
    C = gammma(a + b)
    C /= (gammma(a) * gammma(b))
    y = C * (mu ** (a - 1))
    y *= (1 - mu) ** (b - 1)
    return y


def log_beta(mu, a, b):
    C = gammma(a + b)
    C /= (gammma(a) * gammma(b))
    log_C = np.log(C)
    y = (a - 1) * np.log(mu)
    y += (b - 1) * np.log(1 - mu)
    y += log_C
    return y
