"""c3e9.py

Exercise 3.9 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP
from src.kernel_ls import krls, krls_ald, krls_aldp
from src.kernels import gaussian_kernel, polynomial_kernel


if __name__ == '__main__':

    Imax = 150
    N = 7
    ensemble = 1
    mu_klms = .025  #.06
    gamma_c = .99
    data_set = 'mgdata.xlsx'  # https://www.kaggle.com/datasets/arashabbasi/mackeyglass-time-series
    header = ['t', 't-tau', 't+1']
    df = pd.read_excel(data_set, names=header)
    x_t = df['t'].to_numpy()
    x_delayed = np.hstack((np.zeros((1,)), x_t[:-1]))
    kernel_krls = lambda x, y: polynomial_kernel(x, y, .1, 0, 4)
    # KRLS:
    y_rls = krls_aldp(x_delayed, N, kernel_krls, x_t)
    e_krls = x_t - y_rls

    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(x_t, label='Mackey-Glass')
    ax0.plot(y_rls, label='KRLS')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    fig0.savefig('c3figs/e15.eps', bbox_inches='tight')
    plt.show()
    