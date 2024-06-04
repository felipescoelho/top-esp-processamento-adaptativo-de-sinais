"""e4.py

Script for exercise 4 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 4, 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from dist_filters.incremental_lms import ilms1, ilms2


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 20000
    M = 15
    N = 9
    sigma_eta2 = 0.01
    h = np.ones((N+1,))
    h /= np.linalg.norm(h)
    ensemble = 100
    mu1 = 0.0015
    mu2 = 0.0015
    E1_ensemble = np.zeros((K, M, ensemble))
    W1_ensemble = np.zeros((K+1, M, N+1, ensemble))
    E2_ensemble = np.zeros((K, M, ensemble))
    W2_ensemble = np.zeros((K+1, M, N+1, ensemble))
    for it in range(ensemble):
        X = np.zeros((K, M))
        for m in range(M):
            x_m = lfilter(np.array((1, 2, 1))/np.sqrt(6), [1],
                          rng.standard_normal((K,)))
            X[:, m] = x_m/np.sqrt(np.mean(np.abs(x_m)))
        D = np.array([
            lfilter(h, [1], X[:, m])
            + np.sqrt(sigma_eta2)*rng.standard_normal((K,)) for m in range(M)
        ]).T
        _, E1_ensemble[:, :, it], W1_ensemble[:, :, :, it] = ilms1(X, D, N, mu1)
        _, E2_ensemble[:, :, it], W2_ensemble[:, :, :, it] = ilms2(X, D, N, mu2)
    msE1 = np.mean(E1_ensemble**2, axis=2)
    msE2 = np.mean(E2_ensemble**2, axis=2)
    W1_avg = np.mean(W1_ensemble, axis=3)
    W2_avg = np.mean(W2_ensemble, axis=3)
    msd1 = np.zeros((K, M))
    msd2 = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            msd1[k, m] = np.linalg.norm(
                W1_avg[k+1, m, :]/np.linalg.norm(W1_avg[k+1, m, :]) - h 
            )
            msd2[k, m] = np.linalg.norm(
                W2_avg[k+1, m, :]/np.linalg.norm(W2_avg[k+1, m, :]) - h
            )

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.plot(10*np.log10(msE1[:, -1]))
    ax1.set_ylabel('MSE, dB')
    ax1.set_xlabel('Amostras, $k$')
    ax2 = fig1.add_subplot(212)
    for m in range(M):
        ax2.plot(msd1[1:, m], label=f'Nó {m+1}')
    ax2.legend(ncol=5)
    ax2.set_xlabel('Amostras, $k$')
    ax2.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||$')
    fig1.tight_layout()
    fig1.savefig('c4figs/e4_4_51.eps', bbox_inches='tight')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211)
    ax1.plot(10*np.log10(msE2[:, -1]))
    ax1.set_ylabel('MSE, dB')
    ax1.set_xlabel('Amostras, $k$')
    ax2 = fig2.add_subplot(212)
    for m in range(M):
        ax2.plot(msd2[1:, m], label=f'Nó {m+1}')
    ax2.legend(ncol=5)
    ax2.set_xlabel('Amostras, $k$')
    ax2.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||$')
    fig2.tight_layout()
    fig1.savefig('c4figs/e4_4_53.eps', bbox_inches='tight')

    plt.show()
