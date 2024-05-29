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
from src.kernel_ls import krls
from src.kernel_utils import gaussian


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
    klms_kwargs = {
        'order': N, 'step_factor': mu_klms,
        'kernel_args': (.1, 0, 4),
        'kernel_kwargs': {'kernel_type': 'poly', 'Imax': Imax,
                            'gamma_c': gamma_c,
                            'data_selection': 'no selection'}}
    kernel_lms = KLMS(**klms_kwargs)
    _, e_lms = kernel_lms.run_batch(x_delayed, x_t)

    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(e_lms**2), label='KLMS')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    plt.show()
    