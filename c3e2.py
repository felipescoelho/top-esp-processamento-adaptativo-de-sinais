"""c3e2.py

Exercise 3.2 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP
from src.kernel_ls import krls, krlsg
from src.kernels import gaussian_kernel


def unknown_system(x: np.ndarray, sigma_n_2: float):
    """Unknown system for exercise 3.2"""
    K = len(x)
    d = np.zeros((K,), dtype=x.dtype)
    x_ext = np.hstack((np.zeros((2,), dtype=x.dtype), x))
    for k in range(K):
        d[k] = -.08*x_ext[k+2] - .15*x_ext[k+1] + .14*x_ext[k] \
            + .055 * x_ext[k+2]**2 + .3*x_ext[k+2]*x[k] - .16*x_ext[k+1]**2 \
            + .14*x_ext[k]**2
    n = np.random.randn(K,)
    n *= np.sqrt(sigma_n_2 / np.mean(n**2))

    return d + n


if __name__ == '__main__':
    # Definitions:
    a = [1, -.95]
    b = [1]
    K = 1500
    Imax = 150
    N = 9
    ensemble = 10
    mu_klms = .125
    mu_kap = .125
    gamma_c = .99
    sigma_x_2 = .7
    sigma_n_2 = .01
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
    kernel_krls = lambda x, y: gaussian_kernel(x, y, 2*np.ones(N + 1))
    mse_klms = np.zeros((K, ensemble), dtype=np.float64)
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_krls = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = np.random.randn(K,)
        x *= np.sqrt(sigma_x_2/np.mean(x**2))
        x = lfilter(b, a, x)
        d = unknown_system(x, sigma_n_2)
        kernel_lms = KLMS(**klms_kwargs)
        _, e_lms = kernel_lms.run_batch(x, d)
        mse_klms[:, it] = e_lms**2
        # KAP:
        kap = KAP(**kap_kwargs)
        _, e_kap = kap.run_batch(x, d)
        mse_kap[:, it] = e_kap**2
        # KRLS:
        y = krlsg(x, N, 150, kernel_krls, d, c=.325)
        e_krls = d - y
        mse_krls[:, it] = e_krls**2
    mse_klms_avg = np.mean(mse_klms, axis=1)
    mse_kap_avg = np.mean(mse_kap, axis=1)
    mse_krls_avg = np.mean(mse_krls, axis=1)
    
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
    fig0.savefig('c3figs/e2.eps', bbox_inches='tight')
    plt.show()
