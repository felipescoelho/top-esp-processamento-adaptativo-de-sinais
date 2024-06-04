"""e6.py

Script for exercise 4 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 4, 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from dist_filters.diffusion_lms import cta_lms, atc_lms


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 500
    A = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 1]])
    N = 3
    M = 6
    sigma_eta2 = 1e-3
    h = np.ones((N+1,))
    h /= np.linalg.norm(h)
    mu = 0.01
    mu_d = 0.01
    ensemble = 1
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
        _, E1_ensemble[:, :, it], W1_ensemble[:, :, :, it] = cta_lms(
            X, D, A, N, mu, mu_d
        )
        _, E2_ensemble[:, :, it], W2_ensemble[:, :, :, it] = atc_lms(
            X, D, A, N, mu, mu_d
        )
    msE1 = np.mean(E1_ensemble**2, axis=2)
    msE2 = np.mean(E2_ensemble**2, axis=2)
    W1_avg = np.mean(W1_ensemble, axis=3)
    W2_avg = np.mean(W2_ensemble, axis=3)
    msd1 = np.zeros((K, M))
    msd2 = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            msd1[k, m] = np.linalg.norm(
                W1_avg[k+1, m, :]  # /np.linalg.norm(W1_avg[k+1, m, :])
                - h 
            )
            msd2[k, m] = np.linalg.norm(
                W2_avg[k+1, m, :]  # /np.linalg.norm(W2_avg[k+1, m, :])
                - h
            )
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    for m in range(M):
        ax1.plot(msd1[1:, m], label=f'Nó {m+1}')
    ax1.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||$')
    ax1.set_xlabel('Amostras, $k$')
    ax2 = fig1.add_subplot(212)
    for m in range(M):
        ax2.plot(msd2[1:, m], label=f'Nó {m+1}')
    ax2.legend(ncol=2)
    ax2.set_xlabel('Amostras, $k$')
    ax2.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||$')
    fig1.tight_layout()
    fig1.savefig('c4figs/e6.eps', bbox_inches='tight')

    plt.show()
