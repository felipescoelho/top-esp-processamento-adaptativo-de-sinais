"""kernel_based.py

Script with kernel-based adaptive filter algorithms.

luizfelipe.coelho@smt.ufrj.br
Mar 28, 2024
"""


import numpy as np
from .kernel_utils import (cosine_similarity, sigmoid, polynomial, gaussian,
                           laplacian)
from numba import njit


kernel_dict = {'cosine similarity': cosine_similarity, 'sigmoid': sigmoid,
               'polynomial': polynomial, 'gaussian': gaussian,
               'laplacian': laplacian}


def k_lms(x: np.ndarray, d: np.ndarray, mu: float, N: int, gamma_c: float,
          Imax: int, kernel: str, args: tuple):
    """Kernel Least Mean Squares (k-LMS) Adaptive Filter
    
    Parameters
    ----------
    
    Returns
    -------

    """

    K = len(x)
    e = np.zeros((K,), dtype=x.dtype)
    g = np.zeros((K,), dtype=x.dtype)
    x_ext = np.hstack((np.zeros((N,), dtype=x.dtype), x))
    L_dict = np.zeros((Imax+1,), dtype=np.int32)
    I = 0
    krnl = kernel_dict[kernel]
    for k in range(K):
        tdl = np.flipud(x_ext[k:N+k+1])
        kernel_values = np.zeros((I+1,), dtype=x.dtype)
        kernel_k = krnl(tdl, tdl, *args)
        for i in range(I):
            l = L_dict[i]
            kernel_values[i] = krnl(np.flipud(x_ext[k-l:N+k+1-l]), tdl, *args)
            g[k] += e[l]*kernel_values[i] + 2*mu*kernel_k
        e[k] = d[k] - g[k]
        if np.max(kernel_values) <= gamma_c:
            l_max = np.argmax(kernel_values)
            I += 1
            if I <= Imax:
                L_dict[I] = k
            else:
                L_dict[l_max] = k
        else:
            e[l_max] += mu*e[k]
    
    return e