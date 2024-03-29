"""kernel_utils.py

Kernel functions necessary to implement the kernel-based adaptive
filters.

luizfelipe.coelho@smt.ufrj.br
Mar 28, 2024
"""


import numpy as np


def cosine_similarity(x0: np.ndarray, x1: np.ndarray):
    """Cosine Similarity Kernel.

    Computes: (x0.T @ x1)/(||x0||_2 ||x1||_2)
    
    Parameters
    ----------
    x0 : np.ndarray
        First array in the comparison.
    x1 : np.ndarray
        Second array in the comparison.
    
    Returns
    -------
    y : float
        Resulting similarity between x0 and x1.
    """

    y = np.dot(x0, x1)/(np.linalg.norm(x0, 2)*np.linalg.norm(x1, 2))

    return y


def sigmoid(x0: np.ndarray, x1: np.ndarray, a: float, b: float):
    """Sigmoid Kernel

    Computes: tanh(a * x0.T @ x1 + b)
    
    Parameters
    ----------
    x0 : np.ndarray
        First array in the comparison.
    x1 : np.ndarray
        Second array in the comparison.
    a : float
        Real constant.
    b : float
        Real constant.
    
    Returns
    -------
    y : float
        Resulting similarity between x0 and x1.
    """

    y = np.tanh(a*np.dot(x0, x1) + b)

    return y


def polynomial(x0: np.ndarray, x1: np.ndarray, a: float, b: float, n: int):
    """Polynomial Kernel

    Computes: (a * x0.T @ x1 + b)**n

    Parameters
    ----------
    x0 : np.ndarray
        First array in the comparison.
    x1 : np.ndarray
        Second array in the comparison.
    a : float
        Real constant.
    b : float
        Real constant. b >= 0 is required to guarantee that the kernel
        matrix is positive definite.
    n : int
        Defines the order (degree) of the polynomial.
    
    Returns
    -------
    y : float
        Resulting similarity between x0 and x1.
    """

    y = (a*np.dot(x0, x1) + b)**n

    return y


def gaussian(x0: np.ndarray, x1: np.ndarray, sigma: np.ndarray):
    """Gaussian Kernel
    
    Computes: e**(-.5 * (x0-x1).T @ diag(sigma)**(-1) @ (x0-x1))

    Parameters
    ----------
    x0 : np.ndarray
        First array in comparison.
    x1 : np.ndarray
        Second array in comparison
    sigma : np.ndarray
        Array containing variance for the exponential funcion.

    Returns
    -------
    y : np.ndarray
        Resulting similarity between x0 and x1
    """

    I = len(x0)-1
    if type(sigma) == type(.1):
        sigma = sigma*np.ones((I+1,))
    y = np.exp(-.5* (x0-x1).reshape(1, I+1) @ np.linalg.inv(np.diagflat(sigma))
               @ (x0-x1).reshape(I+1, 1))
    
    return y


def laplacian(x0: np.ndarray, x1: np.ndarray, sigma: float):
    """Laplacian Kernel
    
    Computes e**(-||x0-x1||/sigma)
    
    Parameters
    ----------
    x0 : np.ndarray
        First array in comparison.
    x1 : np.ndarray
        Second array in comparison
    sigma : float
        Bandwidth of the kernel.

    Returns
    -------
    y : np.ndarray
        Resulting similarity between x0 and x1
    """

    y = np.exp(-np.norm(x0-x1, 1)/sigma)

    return y
