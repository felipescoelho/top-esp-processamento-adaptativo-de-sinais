"""incremental_rls.py

Script for the incremental LMS algorithms.

luizfelipe.coelho@smt.ufrj.br
Jun 4, 2024
"""


import numpy as np


def irls(X: np.ndarray, D: np.ndarray, N: int, lamb: float, W_0=None):
    """
    Incremental RLS algorithm
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    S_Dm = np.eye(N+1, dtype=X.dtype)
    for k in range(K):
        S_Dm = S_Dm/lamb
        w = W[k, 0, :]
        for m in range(M):
            Y[k, m] = np.vdot(w, np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            spam = S_Dm @ np.flipud(X_ext[k:k+N+1, m])
            saussage = np.vdot(np.flipud(X_ext[k:k+N+1, m]), spam)
            W[k+1, m, :] = w + np.conj(E[k, m])*spam/(1+saussage)
            S_Dm = S_Dm \
                - np.outer(spam, np.conj(np.flipud(X_ext[k:k+N+1, m])))@S_Dm \
                / (1 + saussage)
            w = W[k+1, m, :]
        W[k+1, 0, :] = w

    return Y, E, W
