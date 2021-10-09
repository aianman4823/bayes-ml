import numpy as np
from scipy import integrate
# ガウス分布に対する確率密度関数


def gauss(x, mu=0, sigma=1):
    c = 1.0 / np.sqrt(2 * np.pi) / sigma
    idx = - 0.5 * ((x - mu) / sigma) ** 2
    return np.exp(idx) * c


def bernouulli(x, mu=0.5):
    return (mu ** x) * (1 - mu)**(1 - x)


if __name__ == '__main__':
    # 連続的
    mu = 0
    sgm = 1
    lower = -10
    upper = 10
    val, err = integrate.quad(gauss, lower, upper, args=(mu, sgm))
    print(val)

    # 離散的
    x = np.linspace(0, 2, 2)
    y = bernouulli(x)
    val = integrate.cumtrapz(y, x)
    print(val)
