"""distributed_dection.py

Script for distributed detection algorithms.

luizfelipe.coelho@smt.ufrj.br
Jun 5, 2024
"""


import numpy as np


def dist_detection_lms(X: np.ndarray, A: np.ndarray, mu: float, gamma0=None,
                       W0=None):
    """
    Least Mean Squares algorithm for distributed detection.

    Parameters
    ----------
    X : np.ndarray
        Input signal array (M, K)
    A : np.ndarray
        Adjacency matrix to indicate connectivity between nodes (M, M)
    mu : float
        Step size (convergence factor)
    gamma : np.ndarray
        Threshold for each node.
    W_0 : np.ndarray
        Initial coefficients for the filter (N+1, M)
    
    Returns
    -------
    Y_soft : np.ndarray
        Output for soft combination in consensus
    Y_hard : np.ndarray
        Output for hard combination consensus
    """

    M, K = X.shape
    W = np.zeros((M, M), dtype=X.dtype) if W0 is None else W0.copy()
    gamma = .5*np.ones((M,), dtype=X.dtype) if gamma0 is None else gamma0.copy()
    Y_soft = np.zeros((M, K), dtype=X.dtype)
    Y_hard = np.zeros((M, K), dtype=X.dtype)
    d_hat = np.zeros((M,), dtype=X.dtype)
    r_hat = np.zeros((M,), dtype=X.dtype)
    E = np.zeros((M, K), dtype=X.dtype)
    for k in range(K):
        # Soft Combination:
        for m in range(M):
            Nm = np.nonzero(A[m, :])
            spam = np.vdot(W[m, Nm], X[Nm, k])
            Y_soft[m, k] = 0 if spam < gamma[m] else 1
            # Local estimate for the reference:
            d_hat[m] = 0 if W[m, m] * X[m, k] < gamma[m] else 1
        # Hard Combination & Adaptive Weight Update:
        for m in range(M):
            Nm = np.nonzero(A[m, :])
            Y_hard[m, k] = 1 if np.sum(Y_soft[Nm, k]) >= 1 else 0
            # Weight update:
            r_hat[m] = 1 if np.sum(d_hat[Nm]) >= 1 else 0
            E[m, k] = r_hat[m] - Y_soft[m, k]
            W[m, Nm] += mu*E[m, k]*X[Nm, k]
    
    return Y_soft, Y_hard
