"""c3e10.py

Exercise 3.10 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP, SMKAP
from src.kernel_ls import krls
from src.kernel_utils import gaussian


def nonlinear_filter(n: np.ndarray):
    n = n.copy()
    K = len(n)
    n = np.hstack((np.zeros((2,)), n))
    x = np.zeros((K+2,))
    for k in range(K):
        x[k+2] = n[k+2] - x[k+1]*n[k+1] - .3*x[k+1] - .5*x[k]
    
    return x[2:]


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 1500
    Imax = 150
    N = 15
    mu_kap = .0001
    mu_smkap = .0001
    gamma_bar = .5
    gamma_c = .99
    sigma_n2 = 1e-2
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (.1, 0, 2), 'kernel_kwargs':{
                      'kernel_type': 'poly', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': False
                  }}
    smkap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                    'gamma_bar': gamma_bar, 'kernel_args': (.1, 0, 2), 'kernel_kwargs':{
                      'kernel_type': 'poly', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': False
                  }}
    n = np.sqrt(sigma_n2)*rng.standard_normal((K,))
    d = np.array([np.sin(.2*np.pi*k)+n[k] for k in range(K)])
    x = nonlinear_filter(n)
    # KAP:
    kap = KAP(**kap_kwargs)
    _, e_kap = kap.run_batch(x, d)
    # SMKAP:
    smkap = SMKAP(**smkap_kwargs)
    _, e_smkap = smkap.run_batch(x, d)
    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(e_kap, label='KAP')
    ax0.plot(e_smkap, label='SMKAP')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    plt.show()
