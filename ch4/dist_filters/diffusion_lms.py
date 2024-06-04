"""diffusion_lms.py

Script for diffusion algorithms using LMS.

luizfelipe.coelho@smt.ufrj.br
Jun 4, 2024
"""


import numpy as np


def cta_lms(X: np.ndarray, D: np.ndarray, A: np.ndarray, N:int, mu: float,
            mu_d: float, W_0=None):
    """
    Combine-then-Adapt diffusion LMS
    
    Parameters
    ----------
    X : np.ndarray
        Input signal array
    D : np.ndarray
        Desired signal array
    A : np.ndarray
        Adjacency array
    N : int
        Filter order
    mu : float
        Convergence factor
    mu_d : float
        Convergence factor for steepest descent
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    delta = np.zeros((M,))
    for k in range(K):
        for m in range(M):
            Nm = A[:, m] != 0
            # Combine
            w_tilde = np.sum(W[k, Nm, :], axis=0)/np.sum(Nm)
            gamma = 1/(1+np.exp(-delta[m]))
            spam = 1-gamma
            phi = gamma*W[k, m, :] + spam*w_tilde
            # Adapt
            Y[k, m] = np.vdot(phi, np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            saussage = W[k, m, :] - w_tilde
            grad_e = -gamma*spam*(
                np.conj(E[k, m])*np.vdot(saussage, np.flipud(X_ext[k:k+N+1, m]))
                + E[k, m]*np.vdot(np.flipud(X_ext[k:k+N+1, m]), saussage)
            )
            delta[m] -= mu_d*grad_e
            W[k+1, m, :] = phi + mu*E[k, m]*np.flipud(X_ext[k:k+N+1, m])

    return Y, E, W


def atc_lms(X: np.ndarray, D: np.ndarray, A: np.ndarray, N:int, mu: float,
            mu_d: float, W_0=None):
    """
    Adapt-then-Combine diffusion LMS
    
    Parameters
    ----------
    X : np.ndarray
        Input signal array
    D : np.ndarray
        Desired signal array
    A : np.ndarray
        Adjacency array
    N : int
        Filter order
    mu : float
        Convergence factor
    mu_d : float
        Convergence factor for steepest descent
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    delta = np.zeros((M,))
    phi = np.zeros((M, N+1))
    for k in range(K):
        for m in range(M):
            # Adapt
            Y[k, m] = np.vdot(W[k, m, :], np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            phi[m, :] = W[k, m, :] + mu*E[k, m]*np.flipud(X_ext[k:k+N+1, m])
            # Combine
            Nm = A[:, m] != 0
            phi_tilde = np.sum(phi[Nm])/np.sum(Nm)
            gamma = 1/(1+np.exp(-delta[m]))
            spam = 1-gamma
            W[k+1, m, :] = gamma*phi[m, :] + spam*phi_tilde
            saussage = phi[m, :] - phi_tilde
            epsilon = D[k, m] - np.vdot(phi[m, :], np.flipud(X_ext[k:k+N+1, m]))
            grad_epsilon = -gamma*spam*(
                np.conj(epsilon)*np.vdot(saussage, np.flipud(X_ext[k:k+N+1, m]))
                + epsilon*np.vdot(np.flipud(X_ext[k:k+N+1, m]), saussage)
            )
            delta[m] -= mu_d*grad_epsilon

    return Y, E, W
