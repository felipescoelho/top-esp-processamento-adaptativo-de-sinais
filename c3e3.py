"""c3e3.py

Exercise 3.3 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP
from src.kernel_ls import krls
from src.kernel_utils import gaussian


def unknown_system(x: np.ndarray, sigma_n_2: float):
    """Unknown system for exercise 3.1"""
    K = len(x)
    d = np.zeros((K,))
    x_ext = np.hstack((np.zeros((2,)), x))
    for k in range(K):
        d[k] = -.76*x_ext[k+2] - 1.0*x_ext[k+1] + 1.0*x_ext[k] \
            + .5 * x_ext[k+2]**2 + 2.0*x_ext[k+2]*x_ext[k] \
            - 1.6 * x_ext[k+1]**2 + 1.2 * x_ext[k]**2 \
            + .8*x_ext[k+1]*x_ext[k]
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
    mu_klms = .125  #.06
    gamma_c = .99
    gamma_d = .1
    gamma_e = .1
    sigma_x_2 = .1
    sigma_n_2 = 1e-2
    klms1_kwargs = {'order': N, 'step_factor': mu_klms,
                    'kernel_args': (.2*np.ones((N+1,)),),
                    'kernel_kwargs': {'kernel_type': 'gauss', 'Imax': Imax,
                                      'gamma_c': gamma_c,
                                      'data_selection': 'coherence approach'}}
    klms2_kwargs = {'order': N, 'step_factor': mu_klms,
                    'kernel_args': (.2*np.ones((N+1,)),),
                    'kernel_kwargs': {'kernel_type': 'gauss', 'Imax': Imax,
                                      'gamma_d': gamma_d, 'gamma_e': gamma_e,
                                      'data_selection': 'novelty criterion'}}
    mse_klms1 = np.zeros((K, ensemble), dtype=np.float64)
    mse_klms2 = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = np.random.randn(K,)
        x *= np.sqrt(sigma_x_2/np.mean(x**2))
        x = lfilter(b, a, x)
        d = unknown_system(x, sigma_n_2)
        # KLMS 1:
        klms1 = KLMS(**klms1_kwargs)
        _, e_klms1 = klms1.run_batch(x, d)
        mse_klms1[:, it] = e_klms1**2
        # KLMS 2:
        klms2 = KLMS(**klms2_kwargs)
        _, e_klms2 = klms2.run_batch(x, d)
        mse_klms2[:, it] = e_klms2**2
    mse_klms1_avg = np.mean(mse_klms1, axis=1)
    mse_klms2_avg = np.mean(mse_klms2, axis=1)
    
    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(mse_klms1_avg), label='Coherence Approach')
    ax0.plot(10*np.log10(mse_klms2_avg), label='Novelty Criterion')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()
    fig0.savefig('c3figs/e3.eps', bbox_inches='tight')
    plt.show()
