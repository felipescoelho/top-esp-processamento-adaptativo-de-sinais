#!/usr/bin/python
# -*- coding: utf-8 -*-


#----------------------------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt

import numpy as np


#----------------------------------------------------------------------------------------------#
# MAIN CODE
#----------------------------------------------------------------------------------------------#

'''

    The codes developed here are based on the paper:

    Distributed Algorithms for Array Signal Processing, Po-Chih Chen & P. P. Vaidyanathan

'''

def Wt(N_m, delta_zero = 1e-6, delta_equal = 1e-4):

    '''
        Implementation of the equation 7 and 8.
    '''

    P = len(N_m)

    A = np.array([[1 if i in N else 0 for i in range(P)] for N in N_m])

    L = np.diag(np.sum(A, axis = 0)) - A

    e_values = np.linalg.eig(L)[0]

    e_values[np.abs(e_values) < delta_zero] = 0 # To guarantee the eigenvalue 0

    e_values_unique_indexes = []

    e_values_unique = set()

    for index, value in enumerate(np.round(e_values / delta_equal) * delta_equal):
        if value not in e_values_unique:
            e_values_unique.add(value)
            e_values_unique_indexes.append(index)

    e_values = e_values[e_values_unique_indexes]

    R = len(e_values)

    e_values = e_values[e_values != 0]

    W = np.zeros((R - 1, P, P))

    W[0] = (-1) ** (R - 1) / np.product(e_values) * (L - e_values[0] * np.eye(P))

    for t in range(1, R - 1):
        W[t] = L - e_values[t] * np.eye(P)

    return A, L, R, W


#----------------------------------------------------------------------------------------------#

def AC(W, u):

    '''
        Implementation of the equation 5.
    '''

    for t in range(W.shape[0]):
        u = W[t] @ u

    return u

#----------------------------------------------------------------------------------------------#

def power_iteration_method(x, P, I_max = 100):

    N, K = x.shape

    Q = int(N / P)

    indexes = np.array([range(p * Q, (p + 1) * Q) for p in range(P)])

    R_hat_xx = np.zeros((N, N), dtype = np.complex128)

    for k in range(K):
        R_hat_xx += 1 / K * x[:, k].reshape(N, 1) @ np.conjugate(x[:, k].reshape(N, 1)).T
    ev = np.linalg.eig(R_hat_xx)[1]
    e = np.exp(1j * np.random.random(N))

    for i in range(I_max):

        e_new = 0

        for k in range(K):

            b_k = 0

            for p in range(P):
                b_k += np.dot(np.conjugate(x[indexes[p], k]), e[indexes[p]])
                print(np.dot(np.conjugate(x[indexes[p], k]), e[indexes[p]]))

            e_new += 1 / K * x[:, k] * b_k

        e = e_new.copy()
        # e = e_new / np.abs(e_new)
        # print(e)

    e /= np.abs(e)

    return e, ev

#----------------------------------------------------------------------------------------------#

def MUSIC(a_bar, a_under, b_bar, b_under, E, I_max, epsilon = 1e-6):

    '''
        E_hat_s_p \in C^(P X Q X D)
    '''

    z_i = np.exp(1j * np.random.random(N))

    # z_i = np.random.random(N) + 1j * np.random.random(N)

    for i in range(I_max):

        for k in range(2 * N):

            u_bar = np.sum(np.dot(E))

        if np.abs(z[i + 1, k] - z[i, k]) ** 2 < epsilon:
            return z

    return z

#----------------------------------------------------------------------------------------------#

def spatial_smoothing(x, P, Q, L_ss, L):

    '''

            Implementation of the Algorithm 4

    '''

    K = x.shape[1]

    t_i = np.zeros((L_ss, K), dtype = x.dtype)

    q_i = np.zeros((L_ss, P), dtype = object)

    q_i_flatten = np.zeros(L_ss, dtype = object)

    e_1 = np.random.random(P * Q)

    e_1_n = e_1.astype(np.complex128)

    s_m_p = np.zeros((L, P), dtype = x.dtype)

    indexes_final = np.zeros((L_ss, P), dtype = object)

    len_q_i_flatten = []

    for i in range(L_ss):

        indexes = np.array(range(i, i + N - L_ss))

        for k in range(K):

            for p in range(P):

                indexes_p = indexes[(indexes >= p * Q) & (indexes < (p + 1) * Q)]

                indexes_final[i, p] = indexes_p

                if indexes_p.size:
                    t_i[i, k] += np.vdot(np.conjugate(x[indexes_p, k]), e_1[indexes_p])

        flatten = []

        for p in range(P):     
            if indexes_final[i, p].size:
                for k in range(K):
                    q_i[i, p] += x[indexes_final[i, p], k] * t_i[i, k] / K

            if isinstance(q_i[i, p], np.ndarray):
                flatten.extend(q_i[i, p])

        q_i_flatten[i] = flatten

        len_q_i_flatten.append(len(flatten))

    for m in range(L):

        for p in range(P):
            for i in range(np.maximum(0, p * Q - m), np.maximum(0, (p + 1) * Q - m - 1)):
                    if i < L_ss and m < len(q_i_flatten[i]):
                        s_m_p[m, p] += q_i_flatten[i][m] / L_ss

        e_1_n[m] = np.sum(s_m_p[m]) 

    return e_1_n


#----------------------------------------------------------------------------------------------#
# MAIN SCRIPT
#----------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    # P = 6

    # N_m = [{1, 2}, {0, 2}, {0, 1, 3}, {2, 4, 5}, {3, 5}, {3, 4}]

    # A, L, R, W = Wt(N_m)

    # Q = 16

    # theta = np.array(range(-10, 14, 4))

    # power = 1

    # L_ss = 17

    # K = 500

    # ensemble = 2000

    '''
        Example from the Simulations section.
    '''

    from scipy.signal import remez

    rng = np.random.default_rng()

    P = 6  # Number of nodes
    Q = 8  # Number of sensors in each node
    N = P * Q  # Number of sensors
    L = 9  # Filter length
    M = 4  # Decimation ratio
    D = 5  # Number of directions

    plot = 0

    doa = [np.pi * np.sin(-np.pi / 36), np.pi * np.sin(np.pi / 36), np.pi / 2, np.pi * 0.74, np.pi * 0.98]  # DOA from simulation parameteres

    A = np.array([[np.exp(1j * doa[d] * n) for d in range(D)] for n in range(N)], dtype = np.complex128)

    ensemble = 2000
    K = 100  # Number of snapshots
    startband = 1 / (4 * M)
    stopband = 3 / (4 * M)
    bands = [0, startband, stopband, 0.5]
    desired = [1, 0]

    h = remez(L, bands, desired)
    adjacency_mat = np.array([[0, 1, 1, 1, 0, 0],
                              [1, 0, 1, 1, 0, 0],
                              [1, 1, 0, 0, 1, 1],
                              [1, 1, 0, 0, 1, 1],
                              [0, 0, 1, 1, 0, 1],
                              [0, 0, 1, 1, 1, 0]], dtype = np.float64)

    degree_mat = np.diag(np.sum(adjacency_mat, axis = 0))

    laplacian_mat = degree_mat - adjacency_mat

    cov_mat = np.array([[10 ** (-0.5), 0           , 0              , 0              , 0              ],
                        [0           , 10 ** (-0.5), 0              , 0              , 0              ],
                        [0           , 0           , 10 ** 1.5      , 0.6 * 10 ** 1.5, 0.6 * 10 ** 1.5],
                        [0           , 0           , 0.6 * 10 ** 1.5, 10 ** 1.5      , 0.6 * 10 ** 1.5],
                        [0           , 0           , 0.6 * 10 ** 1.5, 0.6 * 10 ** 1.5, 10 ** 1.5      ]])

    for it in range(ensemble):

        c = rng.multivariate_normal(np.zeros(D), cov_mat, K).T
        e = rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))

        for k in range(K):
            e[:, k] /= np.sqrt(np.vdot(e[:, k], e[:, k]) / N)

        x = A @ c + e

    if plot:

        # Plot images:
       
        H = np.fft.fftshift(np.fft.fft(h, 2 ** 10))
        f = np.arange(-1, 1, 1/2 ** 9)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.plot(f, 20 * np.log10(np.abs(H)), label = 'Traditional CBS')

        ymin, ymax = ax.get_ylim()

        ax.vlines(2 * startband , ymin, ymax, colors = 'tab:gray', ls = '-.', label = 'Passband edge')
        ax.vlines(-2 * startband, ymin, ymax, colors = 'tab:gray', ls = '-.')
        ax.vlines(2 * stopband  , ymin, ymax, colors = 'tab:red' , ls = '-.', label = 'Stopband edge')
        ax.vlines(-2 * stopband , ymin, ymax, colors = 'tab:red' , ls = '-.')

        for idx, true_doa in enumerate(doa):

            if idx == 0:
                ax.vlines(true_doa/np.pi, ymin, ymax, colors = 'tab:orange', ls = '--', label = 'True DOAs')

            else:
                ax.vlines(true_doa/np.pi, ymin, ymax, colors = 'tab:orange', ls = '--')

        ax.set_xlabel('Normalized Frequency, $\omega/\pi$')
        ax.set_ylabel('Magnitude, dB')

        ax.grid()

        ax.legend()

        ax.set_ylim((ymin, ymax))

        fig.tight_layout()

        plt.show()

    L_ss = 17

    # e = spatial_smoothing(x, P, Q, L_ss, L)

    e, ev = power_iteration_method(x, P)


#----------------------------------------------------------------------------------------------#
# End of File (EOF)
#----------------------------------------------------------------------------------------------#