"""e12.py

Script for exercise 12 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 6, 2024
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dist_filters.distributed_detection import (dist_detection_lms,
                                                detection_lms)


def estimate_pd_pf(x: np.ndarray, true_vect: np.ndarray):
    """
    A method to estimate the probability of detection and false alarm.
    """
    K, N = x.shape
    L = np.sum(true_vect)  # Total number of H1
    M = K-L  # Total number of H0
    pf = np.zeros((N,))
    pd = np.zeros((N,))
    for k in range(K):
        for n in range(N):
            # FP/(FP+TN) same as fall-out
            pf[n] += 1/M if x[k, n] == 1 and x[k, n] != true_vect[k] else 0
            # TP/(TP+FN) same as recall
            pd[n] += 1/L if x[k, n] == 1 and x[k, n] == true_vect[k] else 0
    
    return pd, pf


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = int(1e5)
    M = 12
    mu = 1e-5
    P_fa = np.linspace(0, 1, 25)
    P_fa[0] = 1e-12
    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)
    A += np.eye(M)
    info_rate = rng.random((1,))[0]  # Number of H0 / number of H1 in K
    true_vect = rng.choice((0, 1), (K,), p=(info_rate, 1-info_rate))
    # Generate data for each node:
    X = np.zeros((M, K))
    for k in range(K):
        X[:, k] = (2 + rng.standard_normal((M,)))**2 \
            if true_vect[k] == 1 else rng.standard_normal((M,))**2
    u_soft = np.zeros((M, K, len(P_fa)))
    u_hard = np.zeros((M, K, len(P_fa)))
    u_solo = np.zeros((K, len(P_fa)))
    for idx in tqdm(range(len(P_fa))):
        u_soft[:, :, idx], u_hard[:, :, idx] = dist_detection_lms(X, A, mu,
                                                                  P_fa[idx])
        u_solo[:, idx] = detection_lms(X[0, :], mu, P_fa[idx])
    
    pd_soft_est, pf_soft_est = estimate_pd_pf(u_soft[0, :, :], true_vect)
    pd_hard_est, pf_hard_est = estimate_pd_pf(u_hard[0, :, :], true_vect)
    pd_solo_est, pf_solo_est = estimate_pd_pf(u_solo, true_vect)

    folder_name = 'c4figs/'
    os.makedirs(folder_name, exist_ok=True)
    fig1path = os.path.join(folder_name, 'e12_est.eps')

    mask_solo = 1-pd_solo_est > 1e-5
    mask_soft = 1-pd_soft_est > 1e-5
    mask_hard = 1-pf_hard_est > 1e-5

    fig1 = plt.figure()
    ax0 = fig1.add_subplot(111)
    ax0.plot(pf_solo_est[mask_solo], 1-pd_solo_est[mask_solo], '-d',
             label='Node 1 -- Single')
    ax0.plot(pf_soft_est[mask_soft], 1-pd_soft_est[mask_soft], '-^',
             label='Node 1 -- Soft')
    ax0.plot(pf_hard_est[mask_hard], 1-pd_hard_est[mask_hard], '-s',
             label='Node 1 -- Hard')
    ax0.set_yscale('log', base=10)
    ax0.set_xlabel('$P_f$')
    ax0.set_ylabel('$1 - P_d$')
    # ax0.set_xlim((0, 1))
    ax0.legend()
    ax0.grid()
    fig1.tight_layout()
    fig1.savefig(fig1path, bbox_inches='tight')

    plt.show()
