import os
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
                    mu_truth=None,
                    posterior=None,
                    true_model=None):
    fig = plt.figure(figsize=(12, 9))
    plt.plot(x_line, model, color=color)
    if mu_truth is not None and posterior is not None:
        plt.vlines(x=mu_truth, ymin=0, ymax=max(posterior),
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
