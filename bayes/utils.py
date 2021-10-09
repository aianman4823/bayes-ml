import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def my_makedirs(path):
    if not Path(path).exists():
        os.makedirs(path)


def plot_likelihood(x_line,
                    model,
                    xlabel='x',
                    ylabel='density',
                    color='purple',
                    suptitle='default_suptitle',
                    title='default_title',
                    output_path='../outputs/default',
                    name='default',
                    truth=None,
                    true_model=None):
    fig = plt.figure(figsize=(12, 9))
    plt.plot(x_line, model, color=color)
    if truth is not None:
        plt.vlines(x=truth, ymin=0, ymax=max(model),
                   label='$\mu_{truth}$', color='red', linestyle='--')
    if true_model is not None:
        plt.plot(x_line, true_model, color='red', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(suptitle, fontsize=20)
    plt.title(title, loc='left')
    my_makedirs(output_path)
    fig.savefig(output_path + '/' + name)
    pass


def plot_hist(x,
              bins,
              xlabel='x',
              ylabel='density',
              suptitle='default_suptitle',
              title='default_title',
              output_path='../outputs/default',
              name='default'):
    fig = plt.figure(figsize=(12, 9))
    plt.hist(x, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(suptitle, fontsize=20)
    plt.title(title, loc='left')
    my_makedirs(output_path)
    fig.savefig(output_path + name)
    pass


def make_xline(mu, lmb, n):
    """ガウス分布に従うと仮定しているとき"""
    xline = np.linspace(mu - 4 * np.sqrt(1 / lmb),
                        mu + 4 * np.sqrt(1 / lmb),
                        num=n)
    return xline


def plot_likelihood_2d(x_0_grid,
                       x_1_grid,
                       x_dims,
                       model,
                       xlabel='x',
                       ylabel='density',
                       color='purple',
                       suptitle='default_suptitle',
                       title='default_title',
                       output_path='../outputs/default',
                       name='default',
                       truth=None,
                       true_model=None,
                       x_nd=None):
    fig = plt.figure(figsize=(12, 9))
    plt.contour(x_0_grid, x_1_grid, model.reshape(x_dims))  # 尤度
    if truth is not None:
        plt.scatter(x=truth[0], y=truth[1], color='red', s=100, marker='x')
    if x_nd is not None:
        plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1])  # 観測データ

    if true_model is not None:
        plt.contour(x_0_grid, x_1_grid, true_model.reshape(x_dims), alpha=0.5, linestyles='--')  # 尤度

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(suptitle, fontsize=20)
    plt.title(title, loc='left')
    plt.colorbar()
    my_makedirs(output_path)
    fig.savefig(output_path + '/' + name)
    pass


def plot_scatter_2d(x_nd,
                    x_0_grid,
                    x_1_grid,
                    x_dims,
                    model,
                    xlabel='x',
                    ylabel='density',
                    color='purple',
                    suptitle='default_suptitle',
                    title='default_title',
                    output_path='../outputs/default',
                    name='default.png',
                    truth=None,
                    true_model=None):
    plt.figure(figsize=(12, 9))
    plt.scatter(x=x_nd[:, 0], y=x_nd[:, 1])  # 観測データ
    plt.contour(x_0_grid, x_1_grid, model.reshape(x_dims))  # 真の分布
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(suptitle, fontsize=20)
    plt.title(title, loc='left')
    plt.colorbar()
    my_makedirs(output_path)
    plt.savefig(output_path + name)
