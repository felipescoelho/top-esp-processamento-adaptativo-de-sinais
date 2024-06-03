"""c3e7.py

Exercise 3.7 from OLAF

luizfelipe.coelho@smt.ufrj.br
May 28, 2024
"""


import matplotlib.pyplot as plt
import numpy as np
from src.kernel_based import KAP, SMKAP


def linear_system(x):
    K = len(x)
    y = np.zeros((K+3,))
    xx = np.hstack((np.zeros((3,)), x))
    for k in range(K):
        if k < 1500:
            y[k+3] = .25*xx[k+3] + .25*xx[k+2] + .25*xx[k+1] + .25*xx[k]
        else:
            y[k+3] = .25*xx[k+3] - .25*xx[k+2] + .25*xx[k+1] - .25*xx[k]
    return y[3:]


def non_linear_function(u):
    return (2/(1+np.exp(-u))) - 1


def awgn(y, snr, seed=42):
    rng = np.random.default_rng(seed)
    K = len(y)
    n = rng.standard_normal((K,))
    n *= np.sqrt((np.mean(y**2)*10**(-snr/10)) / np.mean(n**2))
    
    return y+n


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 3000
    Imax = 500
    N = 31
    ensemble = 1
    mu_kap = .25
    mu_smkap = .25
    gamma_bar = .05
    gamma_c = .99
    delay_len = N + 3
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (.2*np.ones((N+1,)),), 'kernel_kwargs':{
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
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_smkap = np.zeros((K, ensemble), dtype=np.float64)
    # Compute:
    for it in range(ensemble):
        x = rng.choice((0., 1.), (K,))
        u = linear_system(x)
        y = non_linear_function(u)
        r = awgn(y, -20, rng.integers(9999999))
        d = np.hstack((np.zeros((delay_len,)), x[:-delay_len]))
        # KAP:
        kap = KAP(**kap_kwargs)
        y_kap, e_kap = kap.run_batch(r, d)
        mse_kap[:, it] = e_kap**2
        # SMKAP:
        smkap = SMKAP(**smkap_kwargs)
        y_smkap, e_smkap = smkap.run_batch(r, d)
        mse_smkap[:, it] = e_smkap**2
    mse_kap_avg = np.mean(mse_kap, axis=1)
    mse_smkap_avg = np.mean(mse_smkap, axis=1)
    d_hat_kap = np.array([1 if val >= 0.5 else 0 for val in y_kap])
    d_hat_smkap = np.array([1 if val >= 0.5 else 0 for val in y_smkap])
    e_count_kap = np.cumsum(np.abs(d_hat_kap - d))
    e_count_smkap = np.cumsum(np.abs(d_hat_smkap - d))

    # Plot:
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(mse_kap_avg), label='KAP')
    ax0.plot(10*np.log10(mse_smkap_avg), label='SMKAP')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('MSE, dB')
    # ax0.set_ylim((-40, 10))
    fig0.tight_layout()
    fig0.savefig('c3figs/e7_mse.eps', bbox_inches='tight')

    fig1 = plt.figure()
    ax0 = fig1.add_subplot(111)
    ax0.plot(e_count_kap, label='KAP')
    ax0.plot(e_count_smkap, label='SMKAP')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Sample, $k$')
    ax0.set_ylabel('error cumsum')
    fig1.tight_layout()
    fig1.savefig('c3figs/e7_error.eps', bbox_inches='tight')

    plt.show()
