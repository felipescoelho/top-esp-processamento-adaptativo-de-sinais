"""eg7.py

From Example 6.7.

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
    mu = 1e-4  # From solution
    mu_n = 2.5*1e-2  # From solution
    beta = .95  # From solution
    ensemble = 2000
    p = 0.2
    e_lms = np.zeros((N, K, ensemble))
    e_nlms = np.zeros((N, K, ensemble))
    e_rls = np.zeros((N, K, ensemble))
    coeff_dev_lms = np.zeros((M+1, K, ensemble))
    coeff_dev_nlms = np.zeros((M+1, K, ensemble))
    coeff_dev_rls = np.zeros((M+1, K, ensemble))
    w_o = rng.standard_normal((M+1, ensemble))

    for it in range(ensemble):
        adj_mat = gen_erdos_renyi(N, p, rng.integers(9999999))  # From solution
        deg_mat = np.diag(np.sum(adj_mat, axis=0))
        lap_mat = deg_mat - adj_mat
        lap_mat /= np.linalg.norm(lap_mat, ord=2)
        X = np.zeros((M+1, N, K))
        D = np.zeros((N, K))
        for k in range(K):
            for m in range(M+1):
                X[m, :, k] = np.linalg.matrix_power(lap_mat, m) \
                    @ (np.sqrt(rng.random()+.5)*rng.standard_normal((N,)))
            D[:, k] = X[:, :, k].T @ np.conj(w_o[:, it]) \
                + np.sqrt(.1*rng.random()+.05)*rng.standard_normal((N,))
        _, e_lms[:, :, it], W_lms = glms(X, D, M, N, mu)
        _, e_nlms[:, :, it], W_nlms = gnlms(X, D, M, N, mu_n)
        _, e_rls[:, :, it], W_rls = grls(X, D, M, N, beta)
        coeff_dev_lms[:, :, it] = np.tile(w_o[:, it], (K, 1)).T - W_lms[:, 1:]
        coeff_dev_nlms[:, :, it] = np.tile(w_o[:, it], (K, 1)).T - W_nlms[:, 1:]
        coeff_dev_rls[:, :, it] = np.tile(w_o[:, it], (K, 1)).T - W_rls[:, 1:]

    spam = np.mean(e_lms, axis=0)
    eggs = np.mean(e_nlms, axis=0)
    saussage = np.mean(e_rls, axis=0)
    mse_lms = np.mean(spam**2, axis=1)
    mse_nlms = np.mean(eggs**2, axis=1)
    mse_rls = np.mean(saussage**2, axis=1)

    msd_lms = np.mean(np.mean(coeff_dev_lms, axis=2)**2, axis=0)
    msd_nlms = np.mean(np.mean(coeff_dev_nlms, axis=2)**2, axis=0)
    msd_rls = np.mean(np.mean(coeff_dev_rls, axis=2)**2, axis=0)

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

    fig0.savefig('c6figs/eg7_mse.eps', bbox_inches='tight')
    fig1.savefig('c6figs/eg7_msd.eps', bbox_inches='tight')

    plt.show()
        