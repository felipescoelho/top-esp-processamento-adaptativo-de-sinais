"""e9.py

Script for exercise 9 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 6, 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from dist_filters.diffusion_nlms import atc_sm_nlms_sic, atc_sm_nlms_sic_rfb


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 250
    M = 10
    N = 3
    A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    P = A + np.eye(M)
    P = np.diag(np.array([1/Nm for Nm in np.sum(P, axis=0)])) @ P
    sigma_eta2 = np.array((.06, .05, .06, .07, .01, .03, .06, .09, .03, .05))
    a = np.array((1, -.707))
    b = np.array((1,))
    h = np.ones((N+1,))
    h /= np.linalg.norm(h)
    gamma_bar = 0.001
    ensemble = 100
    E1_ensemble = np.zeros((K, M, ensemble))
    W1_ensemble = np.zeros((K+1, M, N+1, ensemble))
    E2_ensemble = np.zeros((K, M, ensemble))
    W2_ensemble = np.zeros((K+1, M, N+1, ensemble))
    for it in range(ensemble):
        X = np.zeros((K, M))
        for m in range(M):
            x_m = lfilter(b, a, rng.standard_normal((K,)))
            X[:, m] = x_m/np.sqrt(np.mean(np.abs(x_m)))
        D = np.array([
            lfilter(h, [1], X[:, m])
            + np.sqrt(sigma_eta2[m])*rng.standard_normal((K,)) for m in range(M)
        ]).T
        _, E1_ensemble[:, :, it], W1_ensemble[:, :, :, it] = atc_sm_nlms_sic(
            X, D, P, N, gamma_bar, agg_mode='priors'
        )
        _, E2_ensemble[:, :, it], W2_ensemble[:, :, :, it] = atc_sm_nlms_sic_rfb(
            X, D, P, N, gamma_bar, agg_mode='priors'
        )
    msE1 = np.mean(E1_ensemble**2, axis=2)
    msE2 = np.mean(E2_ensemble**2, axis=2)
    W1_avg = np.mean(W1_ensemble, axis=3)
    W2_avg = np.mean(W2_ensemble, axis=3)
    msd1 = np.zeros((K, M))
    msd2 = np.zeros((K, M))
    for k in range(K):
        for m in range(M):
            msd1[k, m] = np.linalg.norm(W1_avg[k+1, m, :] - h)
            msd2[k, m] = np.linalg.norm(W2_avg[k+1, m, :] - h)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for m in range(M):
        ax1.plot(20*np.log10(msd1[:, m]), label=f'Nó {m+1}')
    ax1.legend(ncol=2)
    ax1.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||^2$')
    ax1.set_xlabel('Amostras, $k$')
    fig1.tight_layout()

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)
    for m in range(M):
        ax1.plot(20*np.log10(msd2[:, m]), label=f'Nó {m+1}')
    ax1.legend(ncol=2)
    ax1.set_xlabel('Amostras, $k$')
    ax1.set_ylabel('$||{\\bf w}_o - {\\bf w}_m(k)||^2$')
    fig2.tight_layout()

    fig1.savefig('c4figs/e9_cta_lms.eps', bbox_inches='tight')
    fig2.savefig('c4figs/e9_atc_lms.eps', bbox_inches='tight')

    plt.show()
