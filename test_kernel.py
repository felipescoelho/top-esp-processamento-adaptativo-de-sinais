"""test_kernel.py

Script to test kernel adaptive filters using example from the KRLS
paper.

luizfelipe.coelho@smt.ufrj
Apr 29, 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from src.kernel_based import KLMS, KAP


def unknown_system(s: np.ndarray):
    """
    The linear channel is:
        H_1(z) = 1 + .0668z^{-1} - .4764z^{-2} + .8070z^{-3}
    and after 500 symbols it is changed to:
        H_2(z) = 1 - .4326z^{-1} - .6656z^{-2} + .7153z^{-3}.
    A binary signal is sent through this channel and then the nonlinear
    function y = tanh(x) is applied on it, where x is linear channel
    output. Finally, white Gaussian noise is added to match an SNR of
    20 dB.

    Parameters
    ----------
    s : np.ndarray
        Input binary signal.
    
    Returns
    -------
    z : np.ndarray
        Output signal.
    """

    K = len(s)
    snr = 20  # Signal to noise ration in dB
    x = np.zeros((K,), dtype=np.float64)
    s_ext = np.hstack((np.zeros((3,), dtype=np.float64), s))
    for idx in range(K):
        if idx < 500:
            x[idx] = s_ext[idx+3] + .668*s_ext[idx+2] - .4764*s_ext[idx+1] \
                + .8070*s_ext[idx]
        else:
            x[idx] = s_ext[idx+3] - .4326*s_ext[idx+2] - .6656*s_ext[idx+1] \
                + .7153*s_ext[idx]
    y = np.tanh(x)
    noise = np.random.randn(K)
    noise *= np.sqrt((np.mean(y**2)*10**(-snr/10)) / np.mean(noise**2))

    return y+noise


if __name__ == '__main__':
    K = 1500
    N = 15
    mu_klms = .6
    mu_kap = .00002
    gamma_c = .99
    Imax = 400
    ensemble = 5
    klms_kwargs = {'order': N, 'step_factor': mu_klms,
                   'kernel_args': (2*np.ones((N+1,)),), 'kernel_kwargs': {
                       'kernel_type': 'gauss', 'Imax': Imax, 'gamma_c': gamma_c,
                       'data_selection': 'coherence approach'
                   }}
    kap_kwargs = {'order': N, 'step_factor': mu_kap, 'L': 1, 'gamma': 1e-6,
                  'kernel_args': (2*np.ones((N+1,)),), 'kernel_kwargs':{
                      'kernel_type': 'gauss', 'Imax': Imax, 'gamma_c': gamma_c,
                      'data_selection': 'coherence approach',
                      'dict_update': True
                  }}
    mse_klms = np.zeros((K, ensemble), dtype=np.float64)
    mse_kap = np.zeros((K, ensemble), dtype=np.float64)
    mse_krls = np.zeros((K, ensemble), dtype=np.float64)
    for it in range(ensemble):
        s = np.random.randint(2, size=(K,)).astype(np.float64)
        z = unknown_system(s)
        # KLMS:
        # klms = KLMS(**klms_kwargs)
        # _, e = klms.run_batch(s, z)
        # mse_klms[:, it] = e**2
        # KAP:
        kap = KAP(**kap_kwargs)
        _, e = kap.run_batch(s, z)
        mse_kap[:, it] = e**2
        # KRLS:
        # krls = KRLS(**krls_kwargs)
        # _, e = krls.run_batch(s, z)
        # mse_krls[:, it] = e**2
    mse_klms_avg = np.mean(mse_klms, axis=1)
    mse_kap_avg = np.mean(mse_kap, axis=1)
    mse_krls_avg = np.mean(mse_krls, axis=1)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(s, label='Tx Signal')
    ax0.plot(z, label='Rx Signal')
    ax0.set_xlabel('Sample, $n$')
    ax0.set_ylabel('Value')
    ax0.legend()
    fig0.tight_layout()

    fig1 = plt.figure()
    ax0 = fig1.add_subplot(111)
    ax0.plot(10*np.log10(mse_klms_avg), label='KLMS')
    ax0.plot(10*np.log10(mse_kap_avg), label='KAP')
    ax0.plot(10*np.log10(mse_krls_avg), label='KRLS')
    ax0.grid()
    ax0.legend()
    ax0.set_xlabel('Samples, $n$')
    ax0.set_ylabel('MSE, dB')
    fig1.tight_layout()

    plt.show()