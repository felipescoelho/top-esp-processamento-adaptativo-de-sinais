"""tml_based.py

Script for the graph adaptive filters using tapped memory line (TML)

luizfelipe.coelho@smt.ufrj.br
Jun 12, 2024
"""


import numpy as np
from numba import njit


@njit()
def glms(X: np.ndarray, D: np.ndarray, M: int, N: int, mu: float, w_0=None):
    """
    TML-based Graph Least Mean Squares Algorithm

    Parameters
    ----------
    X : np.ndarray
        Input signal (K, M+1, N)
    D : np.ndarray
        Desired Signal (K, N)
    M : int
        Filter order
    N : int
        Number of nodes
    mu : float
        Convergence factor
    w_0 : np.ndarray
        Initial Coefficients

    Returns
    -------
    Y : np.ndarray
        Output signal
    E : np.ndarray
        Error signal
    W : np.ndarray
        Coefficient matrix
    """

    _, _, K = X.shape
    Y = np.zeros((N, K), dtype=X.dtype)
    E = np.zeros((N, K), dtype=X.dtype)
    W = np.zeros((M+1, K+1), dtype=X.dtype)
    W[:, 0] = w_0 if w_0 is not None else np.zeros((M+1,), dtype=X.dtype)
    for k in range(K):
        Y[:, k] = X[:, :, k].T@np.conj(W[:, k])
        E[:, k] = D[:, k] - Y[:, k]
        W[:, k+1] += mu*X[:, :, k]@np.conj(E[:, k])
    
    return Y, E, W


@njit()
def gnlms(X: np.ndarray, D: np.ndarray, M: int, N: int, mu: float, w_0=None):
    """
    TML-based Graph Normalized Least Mean Squares Algorithm

    Parameters
    ----------
    X : np.ndarray
        Input signal
    d : np.ndarray
        Desired Signal
    M : int
        Filter order
    N : int
        Number of nodes
    mu : float
        Convergence factor
    w_0 : np.ndarray
        Initial Coefficients

    Returns
    -------
    Y : np.ndarray
        Output signal
    E : np.ndarray
        Error signal
    W : np.ndarray
        Coefficient matrix
    """

    _, _, K = X.shape
    Y = np.zeros((N, K), dtype=X.dtype)
    E = np.zeros((N, K), dtype=X.dtype)
    W = np.zeros((M+1, K+1), dtype=X.dtype)
    W[:, 0] = w_0 if w_0 is not None else np.zeros((M+1,), dtype=X.dtype)
    if N >= M+1:
        for k in range(K):
            Y[:, k] = X[:, :, k].T @ np.conj(W[:, k])
            E[:, k] = D[:, k] - Y[:, k]
            W[:, k+1] += mu*np.linalg.pinv(X[:, :, k]@np.conj(X[:, :, k]).T) \
                @ X[:, :, k] @ np.conj(E[:, k])
    else:
        for k in range(K):
            Y[:, k] = X[:, :, k].T @ np.conj(W[:, k])
            E[:, k] = D[:, k] - Y[:, k]
            W[:, k+1] += mu * X[:, :, k] \
                @ np.linalg.pinv(np.conj(X[:, :, k]).T @ X[:, :, k]) \
                @ np.conj(E[:, k])
    
    return Y, E, W


@njit()
def grls(X: np.ndarray, D: np.ndarray, M: int, N: int, beta: float, w_0=None):
    """
    TML-based Graph Normalized Least Mean Squares Algorithm

    Parameters
    ----------
    X : np.ndarray
        Input signal
    d : np.ndarray
        Desired Signal
    M : int
        Filter order
    N : int
        Number of nodes
    beta : float
        Forgenting factor
    w_0 : np.ndarray
        Initial Coefficients

    Returns
    -------
    Y : np.ndarray
        Output signal
    E : np.ndarray
        Error signal
    W : np.ndarray
        Coefficient matrix
    """

    _, _, K = X.shape
    Y = np.zeros((N, K), dtype=X.dtype)
    E = np.zeros((N, K), dtype=X.dtype)
    W = np.zeros((M+1, K+1), dtype=X.dtype)
    W[:, 0] = w_0 if w_0 is not None else np.zeros((M+1,), dtype=X.dtype)
    R = np.zeros((M+1, M+1), dtype=X.dtype)
    for k in range(K):
        Y[:, k] = X[:, :, k].T @ np.conj(W[:, k])
        E[:, k] = D[:, k] - Y[:, k]
        R = beta*R + X[:, :, k] @ np.conj(X[:, :, k]).T
        W[:, k+1] += np.linalg.pinv(R)@X[:, :, k]@np.conj(E[:, k])
    
    return Y, E, W

