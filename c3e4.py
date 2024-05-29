"""c3e4.py

Exercise 3.4 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS, KAP


def non_linear_function(u):
    return (2/1+np.exp(-u)) - 1


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
    Imax = 150
    N = 9
    ensemble = 1
    mu_klms = .25  #.06
    mu_kap = .04
    gamma_c = .99
    sigma_x_2 = .01
    sigma_n_2 = .7
    delay_len = 13
    klms_kwargs = {
        'order': N, 'step_factor': mu_klms,
        'kernel_args': (.2*np.ones((N+1,)),),
        'kernel_kwargs': {'kernel_type': 'gauss', 'Imax': Imax,
                            'gamma_c': gamma_c,
                            'data_selection': 'no selection'}}
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (.2*np.ones((N+1,)),), 'kernel_kwargs':{
                      'kernel_type': 'gauss', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': True
                  }}
    mse_klms = np.zeros((K, ensemble), dtype=np.float64)
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_krls = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = rng.choice((-1., 1.), (K,))
        u = lfilter(b, a, x)
        y = non_linear_function(u)
        r = awgn(y, -20, rng.integers(9999999))
        d = np.hstack((np.zeros((delay_len,)), x[:-delay_len]))
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
