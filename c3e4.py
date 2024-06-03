"""c3e4.py

Exercise 3.4 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP
from src.kernel_ls import krls
from src.kernels import gaussian_kernel


def non_linear_function(u):
    return (2 / (1+np.exp(-u))) - 1


def awgn(y, snr, seed=42):
    rng = np.random.default_rng(seed)
    K = len(y)
    n = rng.standard_normal((K,))
    n *= np.sqrt((np.mean(y**2)*10**(-snr/10)) / np.mean(n**2))
    
    return y+n


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 1500
    a = [1]
    b = [.25, .25, .25, .25]
    Imax = 250
    N = 15
    ensemble = 1
    mu_klms = .25
    mu_kap = .25
    gamma_c = .99
    delay_len = N+3
    klms_kwargs = {'order': N, 'step_factor': mu_klms,
                   'kernel_args': (2*np.ones((N+1,)),),
                   'kernel_kwargs': {'kernel_type': 'gauss', 'Imax': Imax,
                                     'gamma_c': gamma_c,
                                     'data_selection': 'coherence approach'}}
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (2*np.ones((N+1,)),), 'kernel_kwargs':{
                      'kernel_type': 'gauss', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': True
                  }}
    kernel_krls = lambda x, y: gaussian_kernel(x, y, .2*np.ones(N+1))
    mse_klms = np.zeros((K, ensemble), dtype=np.float64)
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_krls = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = rng.choice((0., 1.), (K,))
        u = lfilter(b, a, x)
        y = non_linear_function(u)
        r = awgn(y, -20, rng.integers(9999999))
        d = np.hstack((np.zeros((delay_len,)), x[:-delay_len]))
        # KLMS:
        kernel_lms = KLMS(**klms_kwargs)
        y_lms, e_lms = kernel_lms.run_batch(r, d)
        mse_klms[:, it] = e_lms**2
        # KAP:
        kap = KAP(**kap_kwargs)
        y_kap, e_kap = kap.run_batch(r, d)
        mse_kap[:, it] = e_kap**2
        # KRLS:
        y_rls = krls(r, N, Imax, kernel_krls, d, c=.325)
        e_krls = d - y_rls
        mse_krls[:, it] = e_krls**2
    mse_klms_avg = np.mean(mse_klms, axis=1)
    mse_kap_avg = np.mean(mse_kap, axis=1)
    mse_krls_avg = np.mean(mse_krls, axis=1)

    d_hat_klms = np.array([1 if val >= 0 else -1 for val in y_lms])
    d_hat_kap = np.array([1 if val >= 0 else -1 for val in y_kap])
    d_hat_krls = np.array([1 if val >= 0 else -1 for val in y_rls])

    e_count_klms = np.cumsum(np.abs(d_hat_klms - d)/2)
    e_count_kap = np.cumsum(np.abs(d_hat_kap - d)/2)
    e_count_krls = np.cumsum(np.abs(d_hat_krls - d)/2)

    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(mse_klms_avg), label='KLMS')
    ax0.plot(10*np.log10(mse_kap_avg), label='KAP')
    ax0.plot(10*np.log10(mse_krls_avg), label='KRLS')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    fig0.savefig('c3figs/e4_mse.eps', bbox_inches='tight')

    fig1 = plt.figure()
    ax0 = fig1.add_subplot(111)
    ax0.plot(e_count_klms, label='KLMS')
    ax0.plot(e_count_kap, label='KAP')
    ax0.plot(e_count_krls, label='KRLS')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('error cumsum')
    fig1.tight_layout()
    fig1.savefig('c3figs/e4_error.eps', bbox_inches='tight')

    plt.show()
