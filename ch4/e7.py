"""e7.py

Script for exercise 7 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 5, 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from dist_filters.diffusion_lms import cta_lms1, atc_lms1


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 100000
    N = 3
    M = 3
    A = np.ones((M, M)) - np.eye(M)
    sigma_eta2 = (1, 1, 1e-3)
    h = np.ones((N+1,))
    mu = 0.01
    mu_delta = 10
    ensemble = 100
    W1_ensemble = np.zeros((K+1, M, N+1, ensemble))
    W2_ensemble = np.zeros((K+1, M, N+1, ensemble))
    gamma1_ensemble = np.zeros((K, M, ensemble))
    gamma2_ensemble = np.zeros((K, M, ensemble))
    for it in range(ensemble):
        X = np.zeros((K, M))
        for m in range(M):
            x_m = lfilter(np.array((1, 2, 1))/np.sqrt(6), [1],
                          rng.standard_normal((K,)))
            X[:, m] = x_m/np.sqrt(np.mean(np.abs(x_m)**2))
        D = np.array([
            lfilter(h, [1], X[:, m])
            + np.sqrt(sigma_eta2[m])*rng.standard_normal((K,)) for m in range(M)
        ]).T
        _, _, W1_ensemble[:, :, :, it], gamma1_ensemble[:, :, it] = cta_lms1(
            X, D, A, N, mu, mu_delta
        )
        _, _, W2_ensemble[:, :, :, it], gamma2_ensemble[:, :, it] = atc_lms1(
            X, D, A, N, mu, mu_delta
        )

    gamma1_avg = np.mean(gamma1_ensemble, axis=2)
    gamma2_avg = np.mean(gamma2_ensemble, axis=2)
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
        ax1.plot(20*np.log10(msd1[:, m]), label=f'Nó {m+1}')
    ax1.legend(ncol=2)
    ax1.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||^2$, dB')
    ax1.set_xlabel('Amostras, $k$')
    ax2 = fig1.add_subplot(212)
    for m in range(M):
        ax2.plot(gamma1_avg[:, m], label=f'Nó {m+1}')
    ax2.legend(ncol=2)
    ax2.set_ylabel('$\gamma_m(k)$')
    ax2.set_xlabel('Amostras, $k$')
    fig1.tight_layout()

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211)
    for m in range(M):
        ax1.plot(20*np.log10(msd2[:, m]), label=f'Nó {m+1}')
    ax1.legend(ncol=2)
    ax1.set_xlabel('Amostras, $k$')
    ax1.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||^2$, dB')
    ax2 = fig2.add_subplot(212)
    for m in range(M):
        ax2.plot(gamma2_avg[:, m], label=f'Nó {m+1}')
    ax2.legend(ncol=2)
    ax2.set_ylabel('$\gamma_m(k)$')
    ax2.set_xlabel('Amostras, $k$')
    fig2.tight_layout()

    fig1.savefig('c4figs/e7_cta_lms.eps', bbox_inches='tight')
    fig2.savefig('c4figs/e7_atc_lms.eps', bbox_inches='tight')

    plt.show()
