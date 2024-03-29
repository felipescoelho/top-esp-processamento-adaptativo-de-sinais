"""adaptive_filters.py

Script with adaptive filter algorithms.

luizfelipe.coelho@smt.ufrj.br
Mar 17, 2024
"""


import numpy as np
from numba import njit


@njit()
def lms(x: np.ndarray, d:np.ndarray, mu:float, w0:np.ndarray):
    """Least Mean Squares (LMS) Adaptive Filter
    
    Parameters
    ----------
    x : np.ndarray
        Input signal as a 1d array.
    d : np.ndarray
        Desired signal as a 1d array.
    mu : float
        Step factor.
    w0 : np.ndarray
        Initial coefficients for the filter as a 1d array.

    Returns
    -------
    y : np.ndarray
        Output signal as a 1d array.
    e : np.ndarray
        Error signal as a 1d array.
    W : np.ndarray
        Coefficient matrix, filter coefficients evolution through
        iterations.
    """

    K = len(x)
    N = len(w0)-1
    y = np.zeros((K,), dtype=x.dtype)
    e = np.zeros((K,), dtype=x.dtype)
    W = np.hstack((w0.reshape(N+1, 1), np.zeros((N+1, K-1), dtype=x.dtype)))
    x_ext = np.hstack((np.zeros((N,), dtype=x.dtype), x))
    for k in range(K-1):
        tdl = np.flipud(x_ext[k:N+k+1])
        y[k] = np.conj(W[0:N+1, k].T) @ tdl
        e[k] = d[k] - y[k]
        W[:, k+1] = W[:, k] + mu*np.conj(e[k])*tdl
    
    return y, e, W


@njit()
def nlms(x: np.ndarray, d: np.ndarray, mu: float, gamma: float, w0: np.ndarray):
    """Normalized Least Mean Squares Adaptive Filter
    
    Parameters
    ----------
    x : np.ndarray
        Input signal as a 1d array.
    d : np.ndarray
        Desired signal as a 1d array.
    mu : float
        Step factor.
    gamma : float
        Small constant to avoid singularity.
    w0 : np.ndarray
        Initial coefficients for the filter as a 1d array.

    Returns
    -------
    y : np.ndarray
        Output signal as a 1d array.
    e : np.ndarray
        Error signal as a 1d array.
    W : np.ndarray
        Coefficient matrix, filter coefficients evolution through
        iterations.
    """

    K = len(x)
    N = len(w0)-1
    y = np.zeros((K,), dtype=x.dtype)
    e = np.zeros((K,), dtype=x.dtype)
    W = np.hstack((w0.reshape(N+1, 1), np.zeros((N+1, K-1), dtype=x.dtype)))
    x_ext = np.hstack((np.zeros((N,), dtype=x.dtype), x))
    for k in range(K-1):
        tdl = np.flipud(x_ext[k:N+k+1])
        y[k] = np.conj(W[0:N+1, k].T) @ tdl
        e[k] = d[k] - y[k]
        W[:, k+1] = W[:, k] + mu*np.conj(e[k])*tdl/(np.conj(tdl.T)@tdl + gamma)
    
    return y, e, W


@njit()
def ap_alg(x: np.ndarray, d: np.ndarray, mu: float, gamma: float, L: int,
           w0: np.ndarray):
    """Affine Projection Algorithm Adaptive Filter.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal as a 1d array.
    d : np.ndarray
        Desired signal as a 1d array.
    mu : float
        Step factor.
    gamma : float
        Small constant to avoid ill conditioning at the matrix
        inversion.
    L : int
        Number of vectors in reuse.
    w0 : np.ndarray
        Initial coefficients for the filter as a 1d array.

    Returns
    -------
    y : np.ndarray
        Output signal as a 1d array.
    e : np.ndarray
        Error signal as a 1d array.
    W : np.ndarray
        Coefficient matrix, filter coefficients evolution through
        iterations.
    """

    K = len(x)
    N = len(w0)-1
    y = np.zeros((K, L+1), dtype=x.dtype)
    e = np.zeros((K, L+1), dtype=x.dtype)
    W = np.hstack((w0.reshape(N+1, 1), np.zeros((N+1, K-1), dtype=x.dtype)))
    X = np.zeros((N+1, L+1), dtype=x.dtype)
    x_ext = np.hstack((np.zeros((N,), dtype=x.dtype), x))
    for k in range(L, K-1):
        for l in range(L+1):
            X[:, l] = np.flipud(x_ext[k-l:N+k+1-l])
        y[k, :] = np.conj(W[0:N+1, k].T) @ X
        e[k, :] = np.flipud(d[k-L:k+1]) - y[k, :]
        W[:, k+1] = W[:, k] \
            + mu*X@np.linalg.pinv(np.conj(X.T)@X + gamma*np.eye(L+1)) \
            @ np.conj(e[k, :])
    
    return y, e, W
