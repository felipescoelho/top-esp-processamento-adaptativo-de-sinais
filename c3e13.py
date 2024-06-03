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
    Imax = 250
    N = 9
    a = .03
    b = .0
    n = 4
    mu_kap = .5
    mu_smkap = .5
    gamma_bar = 3
    gamma_c = .99
    sigma_n2 = 10
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (a, b, n), 'kernel_kwargs':{
                      'kernel_type': 'poly', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': True
                  }}
    smkap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                    'gamma_bar': gamma_bar, 'kernel_args': (a, b, n), 'kernel_kwargs':{
                      'kernel_type': 'poly', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': True
                  }}
    n = rng.standard_normal((K,))
    n *= np.sqrt(sigma_n2/np.mean(np.abs(n)**2))  / 10
    d = np.array([np.sin(.2*np.pi*k)/10 + n[k] for k in range(K)])
    x = nonlinear_filter(n)  # Gera overflow
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

    n_out_kap = e_kap - (d-n)
    n_out_smkap = e_smkap - (d-n)
    print(np.mean(np.abs(n)**2))
    print(np.mean(np.abs(n_out_kap)**2))
    print(np.mean(np.abs(n_out_smkap)**2))

    fig1 = plt.figure()
    ax0 = fig1.add_subplot(111)
    ax0.plot(n)
    ax0.plot(n_out_kap)
    ax0.plot(n_out_smkap)

    plt.show()
