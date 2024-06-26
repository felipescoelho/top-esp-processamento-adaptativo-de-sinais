"""distributed_dection.py

Script for distributed detection algorithms.

luizfelipe.coelho@smt.ufrj.br
Jun 5, 2024
"""


import numpy as np
from scipy.special import erfcinv
from tqdm import tqdm


def dist_detection_lms(X: np.ndarray, A: np.ndarray, mu: float, p_fa: float,
                       gamma0=None, W0=None):
    """
    Least Mean Squares algorithm for distributed detection.

    This algorithm is preparated for exercise 12 of the 4th chapter and
    shoud be revised for other applications.

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
    spam = np.zeros((M,), dtype=X.dtype)
    u_soft = np.zeros((M, K), dtype=X.dtype)
    u_hard = np.zeros((M, K), dtype=X.dtype)
    d_hat = np.zeros((M, K), dtype=X.dtype)
    r_hat = np.zeros((M,), dtype=X.dtype)
    E = np.zeros((M, K), dtype=X.dtype)
    for k in tqdm(range(K), leave=False):
        # Soft Combination:
        for m in range(M):
            Nm = np.nonzero(A[m, :])[0]
            spam[m] = np.dot(W[m, Nm], X[Nm, k])
            u_soft[m, k] = 0 if spam[m] < gamma[m] else 1
            # Local estimate for the reference:
            d_hat[m, k] = 0 if W[m, m] * X[m, k] < gamma[m] else 1
        # Hard Combination & Adaptive Weight Update:
        for m in range(M):
            Nm = np.nonzero(A[m, :])[0]
            u_hard[m, k] = 1 if np.sum(u_soft[Nm, k]) >= 1 else 0
            # Weight update:
            r_hat[m] = 1 if np.sum(d_hat[Nm, k]) >= 1 else 0
            E[m, k] = r_hat[m] - spam[m]
            # Threshold update:
            gamma[m] = erfcinv(p_fa)*np.sqrt(np.dot(W[m, Nm], W[m, Nm]))
            W[m, Nm] += 2*mu*E[m, k]*X[Nm, k]
    
    return u_soft, u_hard


def detection_lms(x: np.ndarray, mu: float, pf: float):
    """"""
    K = len(x)
    u = np.zeros((K,), dtype=x.dtype)
    gamma = .5
    w = 0
    for k in tqdm(range(K), leave=False):
        spam = w*x[k]
        u[k] = 0 if spam < gamma else 1
        e = u[k] - spam
        gamma = erfcinv(pf)*np.sqrt(w**2)
        w += 2*mu*e*x[k]
    
    return u
