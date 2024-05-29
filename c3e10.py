"""c3e10.py

Exercise 3.10 from OLAF

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


def nonlinear_channel(x, sigma_n2, seed=42):
    rng = np.random.default_rng(seed)
    u = x - np.hstack((np.zeros((1,)), x[:-1]))
    d = u + u**3 + np.sqrt(sigma_n2)*rng.standard_normal((len(x),))

    return d


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 1500
    Imax = 150
    N = 15
    ensemble = 1
    mu_klms = .25  #.06
    mu_kap = .00007503125
    gamma_c = .99
    sigma_n2 = .0025
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (.1, 0, 3), 'kernel_kwargs':{
                      'kernel_type': 'poly', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': False
                  }}
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    for it in range(ensemble):
        x = rng.standard_normal((K,))
        d = nonlinear_channel(x, sigma_n2, rng.integers(9999999))
        # KAP:
        kap = KAP(**kap_kwargs)
        _, e_kap = kap.run_batch(x, d)
        mse_kap[:, it] = e_kap**2
    mse_kap_avg = np.mean(mse_kap, axis=1)

    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    # ax0.plot(10*np.log10(mse_klms_avg), label='KLMS')
    ax0.plot(10*np.log10(mse_kap_avg), label='KAP')
    # ax0.plot(10*np.log10(mse_krls_avg), label='KRLS')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    plt.show()