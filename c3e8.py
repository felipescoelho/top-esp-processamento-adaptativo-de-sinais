"""c3e4.py

Exercise 3.6 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP


def unknown_system(x, sigma_n2, seed=42):
    rng = np.random.default_rng(seed)
    K = len(x)
    d = np.zeros((K,))
    d[0] = .25
    for k in range(1, K):
        d[k] = (np.exp(-np.abs(d[k-1])) + x[k])**2
    n = np.sqrt(sigma_n2)*rng.standard_normal((K,))

    return x+n


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 1500
    Imax = 150
    N = 31
    ensemble = 1
    mu_klms = .25  #.06
    mu_kap = .04
    gamma_c = .99
    klms_kwargs = {
        'order': N, 'step_factor': mu_klms,
        'kernel_args': (2*np.ones((N+1,)),),
        'kernel_kwargs': {'kernel_type': 'gauss', 'Imax': Imax,
                            'gamma_c': gamma_c,
                            'data_selection': 'no selection'}}
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (2*np.ones((N+1,)),), 'kernel_kwargs':{
                      'kernel_type': 'gauss', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': True
                  }}
    mse_klms = np.zeros((K, ensemble), dtype=np.float64)
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_krls = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = rng.standard_normal((K,))
        d = unknown_system(x, .15, rng.integers(999999))
        # KLMS:
        kernel_lms = KLMS(**klms_kwargs)
        _, e_lms = kernel_lms.run_batch(x, d)
        mse_klms[:, it] = e_lms**2
        # KAP:
        kap = KAP(**kap_kwargs)
        _, e_kap = kap.run_batch(x, d)
        mse_kap[:, it] = e_kap**2
    mse_klms_avg = np.mean(mse_klms, axis=1)
    mse_kap_avg = np.mean(mse_kap, axis=1)
    
    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(mse_klms_avg), label='KLMS')
    ax0.plot(10*np.log10(mse_kap_avg), label='KAP')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    plt.show()
