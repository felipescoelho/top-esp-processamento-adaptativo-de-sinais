"""doa_algorithms.py

Script with DOA algorithms.

luizfelipe.coelho@smt.ufrj.br
Jun 26, 2024
"""


import numpy as np


def extract_eigvectors(x: np.ndarray, D: int):
    """"""
    _, K = x.shape
    Rxx = x @ np.conj(x.T) / K
    _, eigvectors = np.linalg.eig(Rxx)

    return eigvectors[:, :D]


def bk_estimate(x: np.ndarray, e_qp: np.ndarray):
    """
    Estimate b[k] as described next to Equation (12).

    Parameters
    ----------
    x : np.ndarray
        Input signal at the k-th snapshot
    e_qp : np.ndarray
        Q x P
    """

    N = len(x)
    Q, P = e_qp.shape
    bk = 0
    # Consensus of b[k] Eq. (12)
    for p in range(P):
        bk += np.vdot(x[p*Q:(p+1)*Q], e_qp[:, p])/N
    
    return bk


def e_qp_update(e_qp: np.ndarray, x: np.ndarray):
    """"""
    N, K = x.shape
    Q, P = e_qp.shape
    # Update e_qp
    e_qp_new = np.zeros((Q, P))
    for k in range(K):
        for p in range(P):
            bk = bk_estimate(x[:, k], e_qp)
            e_qp_new[:, p] += x[p*Q:(p+1)*Q, k] * bk
    
    return P*e_qp_new/K


def dist_power_method(x: np.ndarray, D: int, P: int, Ipm: int, seed=42):
    """Distributed Power Method for eigenvector estimation.
    
    Parameters
    ----------
    x : np.ndarray
        Input singal, considers every node.
    D : int
        Number of eigenvectors to be extracted.
    P : int
        Number of nodes.
    Ipm : int
        Number of iterations.
    seed : int
        A number for the RNG.
    """

    rng = np.random.default_rng(seed=seed)
    N, K = x.shape
    Q = int(N/P)
    E = np.zeros((N, D), dtype=np.complex128)
    # for q in range(D):
    #     spam = np.zeros((N,), dtype=np.complex128)
    #     e_qp = rng.standard_normal((Q, P)) + 1j*rng.standard_normal((Q, P))
    #     for _ in range(Ipm):
    #         sausage = 0
    #         for k in range(K):
    #             for p in range(P):
    #                 e_qp[:, p] = e_qp_update(e_qp[:, p], x)
    #                 sausage += np.vdot(x[p*Q:(p+1)*Q, k], e_qp[:, p])
    #             spam += x[:, k]*sausage
    #     E[:, q] = spam
    
    # Based on Thiago's implementation
    idx = np.array([range(p*Q, (p+1)*Q) for p in range(P)])
    eggs = np.exp(1j*2*np.pi*rng.standard_normal((N,)))
    for _ in range(Ipm):
        spam = np.zeros((N,), dtype=np.complex128)
        bk = np.zeros((K,), dtype=np.complex128)
        for k in range(K):
            for p in range(P):
                bk[k] += np.vdot(x[idx[p], k], eggs[idx[p]])
            for p in range(P):
                spam[idx[p]] += x[idx[p], k]*bk[k]/K
        sausage = 0
        for p in range(P):
            sausage += np.vdot(spam[idx[p]], spam[idx[p]])
        spam /= sausage**.5
        eggs = spam
    E[:, 0] = spam
    for q in range(1, D):
        eggs = np.exp(1j*2*np.pi*rng.standard_normal((N,)))
        for _ in range(Ipm):
            spam = np.zeros((N,), dtype=np.complex128)
            bk = np.zeros((K,), dtype=np.complex128)
            for k in range(K):
                for p in range(P):
                    bk[k] += np.vdot(x[idx[p], k], eggs[idx[p]])
                for p in range(P):
                    spam[idx[p]] += x[idx[p], k]*bk[k]/K
            bacon = np.zeros((q,), dtype=np.complex128)
            for i in range(q):
                for p in range(P):
                    bacon[i] += np.vdot(E[idx[p], i], spam[idx[p]])
            for i in range(q):
                for p in range(P):
                    spam[idx[p]] -= E[idx[p], i]*bacon
            sausage = 0
            for p in range(P):
                sausage += np.vdot(spam[idx[p]], spam[idx[p]])
            spam /= sausage**.5
            eggs = spam
        E[:, q] = spam
    
    return E


def dist_root_music(x: np.ndarray, P: int, Imax: int, D: int, seed=42):
    """Method for distributed Root-MUSIC
    
    Parameters
    ----------
    x : np.ndarray
        Input array.
    P : int
        Number of nodes in the processing pool.
    Imax : int
        Number of iterations.
    D : int
        Number of sources.
    seed : int
        Integer for random number generator.
    """

    rng = np.random.default_rng()
    z_new = np.exp(1j*rng.standard_normal((N,)))
    N, K = x.shape
    Q = int(N/P)
    
    for _ in range(Imax):
        z_current = z_new.copy()
        for k in range(N):
            a_tbar = np.array([np.exp(z) for z in range()])
            a_bbar = np.array()
            b_tbar = np.array([np.exp(z) for z in range()])
            b_bbar = np.array()
            u_tbar = E @ a_tbar / N
            u_bbar = E @ a_bbar / N
            v_tbar = E @ b_tbar / N
            v_bbar = E @ b_bbar / N
            s = N*z_current[k]**(N-1) - np.dot(u_bbar, u_tbar)
            t = N*(N-1)
            spam = 0
            for l in range(N):
                spam += 1/z_current[k] - z_current[l] if l != k else 0
            z_new[k] -= 1/(t/s - spam)


def filter_mat(h: np.ndarray, N: int):
    """"""
    L = len(h)
    H = np.zeros((int(N-L+1), N))
    