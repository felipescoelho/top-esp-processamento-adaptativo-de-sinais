"""incremental_lms.py

Script for the incremental LMS algorithms.

luizfelipe.coelho@smt.ufrj.br
Jun 4, 2024
"""


import numpy as np


def ilms1(X: np.ndarray, D: np.ndarray, N: int, mu: float, W_0=None):
    """
    Incremental Least Mean Squares Algorithm.
    Using Equation (4.51).

    Here, each node uses the same filter in every iteration, updating it
    at the end of each cycle.

    Parameters
    ----------
    X : np.ndarray
        Input signal array (K, M).
    D : np.ndarray
        Desired signal array (K, M)
    N : int
        Filter order.
    mu : float
        Step size.
    W_0 : np.ndarray
        Initial coefficient vector (M, N)
    
    Returns
    -------
    Y : np.ndarray
        Output signal (K, M)
    E : np.ndarray
        Error signal (K, M)
    W : np.ndarray
        Coefficient array (K, M, N)
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    for k in range(K):
        w = W[k, 0, :]
        for m in range(M):
            Y[k, m] = np.vdot(w, np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            if m == 0:
                W[k+1, m, :] = w \
                    + mu*np.conj(E[k, m])*np.flipud(X_ext[k:k+N+1, m])/M
            else:
                W[k+1, m, :] = W[k+1, m-1, :] \
                    + mu*np.conj(E[k, m])*np.flipud(X_ext[k:k+N+1, m])/M
            # We don't update w for each m.
        W[k+1, 0, :] = W[k+1, -1, :]

    return Y, E, W


def ilms2(X: np.ndarray, D: np.ndarray, N: int, mu: float, W_0=None):
    """
    Incremental Least Mean Squares Algorithm.
    Using Equation (4.53).

    Here, each node uses filters that consider an update from its
    predescessor; hence, the update is performed during the processing
    cycle and not only when it finishes.

    Parameters
    ----------
    X : np.ndarray
        Input signal array (K, M).
    D : np.ndarray
        Desired signal array (K, M)
    N : int
        Filter order.
    mu : float
        Step size.
    W_0 : np.ndarray
        Initial coefficient vector (M, N)
    
    Returns
    -------
    Y : np.ndarray
        Output signal (K, M)
    E : np.ndarray
        Error signal (K, M)
    W : np.ndarray
        Coefficient array (K, M, N)
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    for k in range(K):
        w = W[k, 0, :]
        for m in range(M):
            Y[k, m] = np.vdot(w, np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            W[k+1, m, :] = w + mu*np.conj(E[k, m])*np.flipud(X_ext[k:k+N+1, m])/M
            w = W[k+1, m, :]  # This is for the next m
        W[k+1, 0, :] = w

    return Y, E, W
