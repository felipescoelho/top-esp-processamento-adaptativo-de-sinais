"""e6.py

6th exercise from chapter 6.

luizfelipe.coelho@smt.ufrj.br
Jun 12, 2024
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from graph_filters.tml_based import glms, gnlms, grls


def gen_erdos_renyi(N: int, p: float, seed=42):
    """Generate Adjacency Matrix for Erdös-Renyi Graph"""

    rng = np.random.default_rng(seed=seed)
    A = np.zeros((N, N))
    for n in range(1, N):
        for m in range(n):
            A[n, m] = rng.choice((0, 1), p=(1-p, p))
    
    return A + A.T


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 5000  # From solution
    M = 2  # From solution
    N = 50
    mu = 1e-5  # From solution
    mu_n = 2.5*1e-2  # From solution
    beta = .95  # From solution
    ensemble = 2000
    p = 0.2
    e_lms = np.zeros((N, K, ensemble))
    e_nlms = np.zeros((N, K, ensemble))
    e_rls = np.zeros((N, K, ensemble))
    W_lms = np.zeros((M+1, K+1, ensemble))
    W_nlms = np.zeros((M+1, K+1, ensemble))
    W_rls = np.zeros((M+1, K+1, ensemble))
    w_o = rng.standard_normal((M+1, ensemble))

    for it in range(ensemble):
        adj_mat = gen_erdos_renyi(N, p, rng.integers(9999999))  # From solution
        x = np.zeros(((N, K)))
        X = np.zeros((M+1, N, K))
        D = np.zeros((N, K))
        for k in range(K):
            x[:, k] = (np.sqrt(rng.random()+.5)*rng.standard_normal((N,)))
            for m in range(M+1):
                X[m, :, k] = np.linalg.matrix_power(adj_mat, m)@x[:, k]
            D[:, k] = X[:, :, k].T @ np.conj(w_o[:, it]) \
                + np.sqrt(.1*rng.random()+.05)*rng.standard_normal((N,))
        _, e_lms[:, :, it], W_lms[:, :, it] = glms(X, D, M, N, mu)
        _, e_nlms[:, :, it], W_nlms[:, :, it] = gnlms(X, D, M, N, mu_n)
        _, e_rls[:, :, it], W_rls[:, :, it] = grls(X, D, M, N, beta)

    spam = np.mean(e_lms**2, axis=0)
    eggs = np.mean(e_nlms**2, axis=0)
    saussage = np.mean(e_rls**2, axis=0)
    mse_lms = np.mean(spam, axis=1)
    mse_nlms = np.mean(eggs, axis=1)
    mse_rls = np.mean(saussage, axis=1)
    # MSD:
    sd_lms = np.zeros((K, ensemble))
    sd_nlms = np.zeros((K, ensemble))
    sd_rls = np.zeros((K, ensemble))
    for it in range(ensemble):
        for k in range(K):
            sd_lms[k ,it] = np.mean((W_lms[:, k+1, it] - w_o[:, it])**2)
            sd_nlms[k, it] = np.mean((W_nlms[:, k+1, it] - w_o[:, it])**2)
            sd_rls[k, it] = np.mean((W_rls[:, k+1, it] - w_o[:, it])**2)
    msd_lms = np.mean(sd_lms, axis=1)
    msd_nlms = np.mean(sd_nlms, axis=1)
    msd_rls = np.mean(sd_rls, axis=1)

    os.makedirs('c6figs/', exist_ok=True)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(10*np.log10(mse_lms), label='GLMS')
    ax0.plot(10*np.log10(mse_nlms), label='GNLMS')
    ax0.plot(10*np.log10(mse_rls), label='GRLS')
    ax0.legend()
    ax0.set_xlabel('Amostra, $k$')
    ax0.set_ylabel('MSE, dB')
    fig0.tight_layout()

    fig1 = plt.figure()
    ax0 = fig1.add_subplot(111)
    ax0.plot(10*np.log10(msd_lms), label='GLMS')
    ax0.plot(10*np.log10(msd_nlms), label='GNLMS')
    ax0.plot(10*np.log10(msd_rls), label='GRLS')
    ax0.legend()
    ax0.set_xlabel('Amostra, $k$')
    ax0.set_ylabel('MSD, dB')
    fig1.tight_layout()

    fig0.savefig('c6figs/e6_mse.eps', bbox_inches='tight')
    fig1.savefig('c6figs/e6_msd.eps', bbox_inches='tight')
    fig0.savefig('c6figs/e6_mse.png', bbox_inches='tight')
    fig1.savefig('c6figs/e6_msd.png', bbox_inches='tight')

    plt.show()
