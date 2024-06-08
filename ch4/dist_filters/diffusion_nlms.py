"""diffusion_nlms.py

Script for diffusion algorithms using NLMS

luizfelipe.coelho@smt.ufrj.br
Jun 5, 2024
"""


import numpy as np


def pw_aggregation_local_estimates(phi: np.ndarray, rho: np.ndarray, m: int,
                                   Nm: np.ndarray):
    """
    Pair-wise aggregation of local estimates at node m.

    Parameters
    ----------
    phi : np.ndarray

    """

    for l in Nm:
        saussage = np.vdot(phi[m, :]-phi[l, :], phi[m, :]-phi[l, :])
        spam = (1 - (rho[m]**2 - rho[l]**2)/saussage) / 2
        lamb = spam if 0 < spam < 1 else 0
        phi[m, :] = (1-lamb)*phi[m, :] + lamb*phi[l, :]
        rho[m] = np.sqrt((1-lamb)*rho[m]**2 + lamb*rho[l]**2
                         - lamb*(1-lamb)*saussage)
    
    return phi[m, :], rho[m]


def atc_nlms(X: np.ndarray, D: np.ndarray, A: np.ndarray, N: int,
             mu: float, W_0=None, agg_mode='pair-wise'):
    """
    Adapt-then-Combine Diffusion Normalized Least Mean Squares Algorithm
    with Full Feedforward data share.

    Parameters
    ----------
    X : np.ndarray
        Input signal array
    D : np.ndarray
        Desired signal array
    A : np.ndarray
        If using pair-wise combination, A is the Adjacency matrix plus
        an Identity matrix. If using priors combination, A is the priors
        matrix
    N : int
        Filter order
    mu : float
        Learning rate
    W_0 : np.nadarray
        Initial coefficient values
    add_mode : str
        Method used for combination between nodes
    
    Returns
    -------
    Y : np.ndarray
        Output signal array
    E : np.ndarray
        Error signal array
    W : np.ndarray
        Coefficient array
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    phi = np.zeros((M, N+1), dtype=X.dtype)
    rho = np.ones((M,), dtype=X.dtype)
    for k in range(K):
        # Adapt
        for m in range(M):
            phi[m, :] = W[k, m, :]
            Y[k, m] = np.vdot(W[k, m, :], X_ext[k:k+N+1, m])
            E[k, m] = D[k, m] - Y[k, m]
            for l in np.nonzero(A[m, :])[0]:
                e = D[k, l] - np.vdot(phi[m, :], np.flipud(X_ext[k:k+N+1, l]))
                phi[m, :] += mu*np.conj(e)*np.flipud(X_ext[k:k+N+1, l]) \
                    / np.vdot(np.flipud(X_ext[k:k+N+1, l]),
                              np.flipud(X_ext[k:k+N+1, l]))
        # Combine
        for m in range(M):
            match agg_mode:
                case 'pair-wise':
                    W[k+1, m, :], rho[m] = pw_aggregation_local_estimates(
                        phi, rho, m, np.nonzero(A[m, :])[0]
                    )
                case 'priors':
                        W[k+1, m, :] = phi.T @ A[m, :]

    return Y, E, W


def atc_nlms_nff(X: np.ndarray, D: np.ndarray, A: np.ndarray, N: int,
             mu: float, W_0=None, agg_mode='pair-wise'):
    """
    Adapt-then-Combine Diffusion Normalized Least Mean Squares Algorithm
    without Feedforward data share.

    Parameters
    ----------
    X : np.ndarray
        Input signal array
    D : np.ndarray
        Desired signal array
    A : np.ndarray
        If using pair-wise combination, A is the Adjacency matrix plus
        an Identity matrix. If using priors combination, A is the priors
        matrix
    N : int
        Filter order
    mu : float
        Learning rate
    W_0 : np.nadarray
        Initial coefficient values
    add_mode : str
        Method used for combination between nodes
    
    Returns
    -------
    Y : np.ndarray
        Output signal array
    E : np.ndarray
        Error signal array
    W : np.ndarray
        Coefficient array
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    phi = np.zeros((M, N+1), dtype=X.dtype)
    rho = np.ones((M,), dtype=X.dtype)
    for k in range(K):
        # Adapt
        for m in range(M):
            phi[m, :] = W[k, m, :]
            Y[k, m] = np.vdot(W[k, m, :], X_ext[k:k+N+1, m])
            E[k, m] = D[k, m] - Y[k, m]
            phi[m, :] += mu*np.conj(E[k, m])*np.flipud(X_ext[k:k+N+1, m]) \
                / np.vdot(np.flipud(X_ext[k:k+N+1, m]),
                            np.flipud(X_ext[k:k+N+1, m]))
        # Combine
        for m in range(M):
            match agg_mode:
                case 'pair-wise':
                    W[k+1, m, :], rho[m] = pw_aggregation_local_estimates(
                        phi, rho, m, np.nonzero(A[m, :])[0]
                    )
                case 'priors':
                        W[k+1, m, :] = phi.T @ A[m, :]

    return Y, E, W


def atc_sm_nlms(X: np.ndarray, D: np.ndarray, A: np.ndarray, N: int,
                gamma_bar: float, W_0=None, agg_mode='pair-wise'):
    """
    Adapt-then-Combine Diffusion Set Membership Normalized Least Mean
    Squares Algorithm

    Parameters
    ----------
    X : np.ndarray
        Input signal array
    D : np.ndarray
        Desired signal array
    A : np.ndarray
        If using pair-wise combination, A is the Adjacency matrix plus
        an Identity matrix. If using priors combination, A is the priors
        matrix
    N : int
        Filter order
    gamma_bar : float
        Set membership parameter
    W_0 : np.nadarray
        Initial coefficient values
    add_mode : str
        Method used for combination between nodes
    
    Returns
    -------
    Y : np.ndarray
        Output signal array
    E : np.ndarray
        Error signal array
    W : np.ndarray
        Coefficient array
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    phi = np.zeros((M, N+1), dtype=X.dtype)
    rho = np.ones((M,), dtype=X.dtype)
    for k in range(K):
        # Adapt
        for m in range(M):
            phi[m, :] = W[k, m, :]
            Y[k, m] = np.vdot(W[k, m, :], X_ext[k:k+N+1, m])
            E[k, m] = D[k, m] - Y[k, m]
            for l in np.nonzero(A[m, :])[0]:
                e = D[k, l] - np.vdot(phi[m, :], np.flipud(X_ext[k:k+N+1, l]))
                if np.abs(e) > gamma_bar:
                    mu = 1 - gamma_bar/np.abs(e)
                    phi[m, :] += mu*np.conj(e)*np.flipud(X_ext[k:k+N+1, l]) \
                        / np.vdot(np.flipud(X_ext[k:k+N+1, l]),
                                  np.flipud(X_ext[k:k+N+1, l]))
        # Combine
        for m in range(M):
            if np.any(np.abs(E[k, np.nonzero(A[m, :])]) > gamma_bar):
                match agg_mode:
                    case 'pair-wise':
                        W[k+1, m, :], rho[m] = pw_aggregation_local_estimates(
                            phi, rho, m, np.nonzero(A[m, :])[0]
                        )
                    case 'priors':
                            W[k+1, m, :] = phi.T @ A[m, :]
            else:
                W[k+1, m, :] = phi[m, :]

    return Y, E, W


def atc_sm_nlms_nff(X: np.ndarray, D: np.ndarray, A: np.ndarray, N: int,
                gamma_bar: float, W_0=None, agg_mode='pair-wise'):
    """
    Adapt-then-Combine Diffusion Set Membership Normalized Least Mean
    Squares No Feedforward Algorithm
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    phi = np.zeros((M, N+1), dtype=X.dtype)
    rho = np.ones((M,), dtype=X.dtype)
    for k in range(K):
        # Adapt
        for m in range(M):
            phi[m, :] = W[k, m, :]
            Y[k, m] = np.vdot(phi[m, :], X_ext[k:k+N+1, m])
            E[k, m] = D[k, m] - Y[k, m]
            if np.abs(E[k, m]) > gamma_bar:
                mu = 1 - gamma_bar/np.abs(E[k, m])
                mu = .1 if mu >= .5 else mu
                phi[m, :] += mu*np.conj(E[k, m])*np.flipud(X_ext[k:k+N+1, m]) \
                    / np.vdot(np.flipud(X_ext[k:k+N+1, m]),
                              np.flipud(X_ext[k:k+N+1, m]))
        # Combine
        for m in range(M):
            if phi[np.nonzero(A[m, :]), :] != W[k, np.nonzero(A[m, :]), :]:
                match agg_mode:
                    case 'pair-wise':
                        W[k+1, m, :], rho[m] = pw_aggregation_local_estimates(
                            phi, rho, m, np.nonzero(A[m, :])[0]
                        )
                    case 'priors':
                            W[k+1, m, :] = phi.T @ A[m, :]
            else:
                W[k+1, m, :] = phi[m, :]

    return Y, E, W


def atc_sm_nlms_sic(X: np.ndarray, D: np.ndarray, A: np.ndarray, N: int,
                    gamma_bar: float, W_0=None, agg_mode='pair-wise'):
    """
    Adapt-then-Combine Diffusion Set Membership Normalized Least Mean
    Squares with Spacial Innovation Check Algorithm.

    Parameters
    ----------
    X: np.ndarray
        
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    phi = np.zeros((M, N+1), dtype=X.dtype)
    rho = np.ones((M,), dtype=X.dtype)
    e_m = np.zeros((M,), dtype=X.dtype)
    for k in range(K):
        # Adapt
        for m in range(M):
            phi[m, :] = W[k, m, :]
            Y[k, m] = np.vdot(W[k, m, :], np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            e_m[m] = E[k, m]
        for m in range(M):
            for l in np.nonzero(A[:, m])[0]:
                if e_m[l] <= gamma_bar:
                    continue
                y_l = np.vdot(phi[m, :], X_ext[k:k+N+1, l])
                e_l = D[k, m] - y_l
                if np.abs(e_l) > gamma_bar:
                    mu = 1 - gamma_bar/np.abs(e_l)
                    phi[m, :] += mu*np.conj(e_l)*X_ext[k:k+N+1, l] \
                        / np.vdot(X_ext[k:k+N+1, l], X_ext[k:k+N+1, l])
        # Combine
        for m in range(M):
            if np.any(np.abs(E[k, np.nonzero(A[m, :])]) > gamma_bar):
                match agg_mode:
                    case 'pair-wise':
                        W[k+1, m, :], rho[m] = pw_aggregation_local_estimates(
                            phi, rho, m, np.nonzero(A[m, :])[0]
                        )
                    case 'priors':
                            W[k+1, m, :] = phi.T @ A[m, :]
            else:
                W[k+1, m, :] = phi[m, :]
    
    return Y, E, W


def atc_sm_nlms_sic_rfb(X: np.ndarray, D: np.ndarray, A: np.ndarray, N: int,
                    gamma_bar: float, W_0=None, agg_mode='pair-wise'):
    """
    Adapt-then-Combine Diffusion Set Membership Normalized Least Mean
    Squares with Spatial Innovation Check and Reduced Feedback Trafic
    Algorithm.
    """

    K, M = X.shape
    W = np.zeros((K+1, M, N+1), dtype=X.dtype)
    if W_0 is not None: W[0, :, :] = W_0
    E = np.zeros((K, M), dtype=X.dtype)
    Y = np.zeros((K, M), dtype=X.dtype)
    X_ext = np.vstack((np.zeros((N, M), dtype=X.dtype), X))
    phi = np.zeros((M, N+1), dtype=X.dtype)
    rho = np.ones((M,), dtype=X.dtype)
    e_m = np.zeros((M,), dtype=X.dtype)
    for k in range(K):
        # Adapt
        for m in range(M):
            phi[m, :] = W[k, m, :]
            Y[k, m] = np.vdot(W[k, m, :], np.flipud(X_ext[k:k+N+1, m]))
            E[k, m] = D[k, m] - Y[k, m]
            e_m[m] = E[k, m]
        for m in range(M):
            for l in np.nonzero(A[:, m])[0]:
                if e_m[l] <= gamma_bar:
                    continue
                y_l = np.vdot(phi[m, :], X_ext[k:k+N+1, l])
                e_l = D[k, m] - y_l
                if np.abs(e_l) > gamma_bar:
                    mu = 1 - gamma_bar/np.abs(e_l)
                    phi[m, :] += mu*np.conj(e_l)*X_ext[k:k+N+1, l] \
                        / np.vdot(X_ext[k:k+N+1, l], X_ext[k:k+N+1, l])
        # Combine
        for m in range(M):
            if np.any(np.abs(E[k, np.nonzero(A[m, :])]) > gamma_bar):
                match agg_mode:
                    case 'pair-wise':
                        W[k+1, m, :], rho[m] = pw_aggregation_local_estimates(
                            phi, rho, m, np.nonzero(A[m, :])[0]
                        )
                    case 'priors':
                            W[k+1, m, :] = phi.T @ A[m, :]
            else:
                W[k+1, m, :] = phi[m, :]
    
    return Y, E, W
