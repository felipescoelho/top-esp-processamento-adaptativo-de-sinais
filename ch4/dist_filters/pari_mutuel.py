"""pari_mutuel.py

Script with A dynamic model for Eisenberg-Gale betting strategy for the
pari-mutel system

luizfelipe.coelho@smt.ufrj.br
Jun 3, 2024
"""


import numpy as np


def pari_mutuel(P: np.ndarray, b: np.ndarray, pi: np.ndarray, K: int,
               epsilon=0.01):
    """
    Parameters
    ----------
    P : np.ndarray
        Prior probability matrix.
    b : np.ndarray
        Individual budgets.
    pi : np.ndarray
        Random vector with ||pi||_1 = 1
    K : int
        Total amount of iterations.
    
    Returns
    -------
    beta : np.ndarray
        Matrix with amout of each bettor for each horse.
    """

    M, N = P.shape
    for _ in range(K):
        for i in range(M):
            idx = np.argmax(P[i, :]/pi)
            pi[idx] += b[i]
    pi /= np.sum(np.abs(pi))
    beta = np.zeros((M, N))
    for i in range(M):
        spam = P[i, :]/pi
        saussage = spam >= np.max(spam)-epsilon
        beta[i, saussage] = b[i]/np.sum(saussage)
    
    return beta, pi