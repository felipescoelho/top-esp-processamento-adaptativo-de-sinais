"""c3e4.py

Exercise 3.6 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP, SMKAP
from src.kernel_ls import krls, krlsg
from src.kernels import gaussian_kernel


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
    N = 4
    ensemble = 1
    mu_klms = .25  #.06
    mu_kap = .025
    mu_smkap = .25
    gamma_bar = .1
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
    smkap_kwargs = {'order': N, 'step_factor': mu_smkap, 'L': 1, 'gamma': 1e-6,
                    'gamma_bar': gamma_bar,
                    'kernel_args': (.2*np.ones((N+1,)),), 'kernel_kwargs':{
                        'kernel_type': 'gauss', 'Imax': Imax,
                        'gamma_c': gamma_c, 'dict_update': True,
                        'data_selection': 'coherence approach'
                    }}
    kernel_krls = lambda x, y: gaussian_kernel(x, y, 2*np.ones(N + 1))
    mse_klms = np.zeros((K, ensemble), dtype=np.float64)
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_smkap = np.zeros((K, ensemble), dtype=np.float64)
    mse_krls = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = rng.standard_normal((K,))
        d = unknown_system(x, .15, rng.integers(999999))
        # KLMS:
        klms = KLMS(**klms_kwargs)
        _, e_klms = klms.run_batch(x, d)
        mse_klms[:, it] = e_klms**2
        # KAP:
        kap = KAP(**kap_kwargs)
        _, e_kap = kap.run_batch(x, d)
        mse_kap[:, it] = e_kap**2
        # SMKAP:
        smkap = SMKAP(**smkap_kwargs)
        y_smkap, e_smkap = smkap.run_batch(x, d)
        mse_smkap[:, it] = e_smkap**2
        # KRLS:
        y_rls = krls(x, N, Imax, kernel_krls, d, c=.325)
        e_krls = d - y_rls
        mse_krls[:, it] = e_krls**2
    mse_klms_avg = np.mean(mse_klms, axis=1)
    mse_kap_avg = np.mean(mse_kap, axis=1)
    mse_krls_avg = np.mean(mse_krls, axis=1)
    mse_smkap_avg = np.mean(mse_smkap, axis=1)
    
    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(mse_klms_avg), label='KLMS')
    ax0.plot(10*np.log10(mse_kap_avg), label='KAP')
    ax0.plot(10*np.log10(mse_krls_avg), label='KRLS')
    ax0.plot(10*np.log10(mse_smkap_avg), label='SMKAP')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    fig0.savefig('c3figs/e8.eps', bbox_inches='tight')
    plt.show()
