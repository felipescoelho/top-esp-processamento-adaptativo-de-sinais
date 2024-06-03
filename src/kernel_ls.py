#!/usr/bin/python
# -*- coding: utf-8 -*-


#----------------------------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------------------------#

import numpy as np


#----------------------------------------------------------------------------------------------#
# MAIN CODE
#----------------------------------------------------------------------------------------------#

def klms(x, N, kernel, d, gamma_c, mu, len_max_data_selected): # 0 <= gamma_c < 1 and 0 < mu << 2

    '''

        Parameters
        ----------

        x                    : array of data;
        N                    : length of the filter (Tapped Delay Line (TDL));
        kernel               : function that evaluate the kernel;
        d                    : array of reference data;
        gamma_c              : threshold of the similarity;
        mu                   : learning rate;
        len_max_data_selected: maximum length of the dictionary; 


        Returns
        -------

        x_est: the estimated vector obtained from x.

    '''

    x = np.concatenate((np.zeros(N), x), dtype = x.dtype) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    data_selected       = np.zeros((len_max_data_selected, N + 1), dtype = x.dtype) # Array with the selected data
    index_data_selected = np.zeros(len_max_data_selected, dtype = int)              # Array with the indices of the selected data
    error_data_selected = np.zeros(len_max_data_selected, dtype = x.dtype)          # Array with the error of the selected data

    len_data_selected = 1 # l_0(0) = {x(0)}

    l_max = 0

    samples = len(d)

    x_est = np.zeros(samples) # The estimated value array (to be populated)

    for k in range(samples):

        x_k = np.flipud(x[k : N + k + 1]) # Tapped delay line (reversed)

        kernel_values = np.array([kernel(data_selected[l], x_k) for l in range(len_data_selected)])

        x_est[k] = 2 * mu * (np.dot(error_data_selected[: len_data_selected], kernel_values))

        kernel_values = np.abs(kernel_values)

        if (abs_max_kernel := kernel_values.max()) <= gamma_c:

            if len_data_selected < len_max_data_selected:

                data_selected[len_data_selected]       = x_k
                index_data_selected[len_data_selected] = k
                error_data_selected[len_data_selected] = (d[k] - x_est[k]) / (1 - 2 * mu * kernel(x_k, x_k))

                len_data_selected += 1

            else:

                abs_max_kernel_indices = np.where(kernel_values == abs_max_kernel)[0]
                l_max                  = abs_max_kernel_indices[np.argmin(np.take(index_data_selected, abs_max_kernel_indices))] # To remove always the oldest one
                # l_max = np.argmax(kernel_values)

                data_selected[l_max]       = x_k
                index_data_selected[l_max] = k
                error_data_selected[l_max] = (d[k] - x_est[k]) / (1 - 2 * mu * kernel(x_k, x_k))

        else:
            error_data_selected[l_max] += mu * (d[k] - x_est[k]) / (1 - 2 * mu * kernel(x_k, x_k))

    return x_est

#----------------------------------------------------------------------------------------------#

def krls(x, N, window_size, kernel, y, c = 0.01):

    '''

        ----------------------------------
        Algorithm from the original paper.
        ----------------------------------

        S1: Initialize K_0 as (1 + c) * I and (K_0) ^ (−1) as I / (1 + c)

        S2: for n = 1, 2, ... do

        S3:    Obtain \hat{K}_{n - 1} out of K_{n - 1}

        S4:    Calculate \hat{K}^{-1}_{n - 1} according to Eq. (12)

        S5:    Obtain K_m according to Eq. (10)

        S6:    Calculate K^{-1}_n according to Eq. (11)

        S7:    Obtain the updated solution \alpha_n = (K^{-1}_{n}) * y_n

        end for

    '''

    samples = len(y)

    x = np.concatenate((np.zeros(N + window_size - 1), x), dtype = x.dtype) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory
    y = np.concatenate((np.zeros(window_size - 1), y), dtype = y.dtype) # Preparing the input: 0 padding the vector.

    K_m     = np.eye(window_size) * (1 + c) # S1: Initializing the input
    K_m_inv = np.eye(window_size) / (1 + c) # S1: Initializing the input

    x_est = np.zeros(samples)

    tdl = lambda n: np.flipud(x[n : n + N + 1]) # Tapped delay line (reversed)

    for n in range(samples): # S2

        x_n = tdl(n + window_size - 1)

        A = K_m[1:, 1:] # S3

        e = K_m_inv[0, 0]
        f = K_m_inv[1:, 0]
        G = K_m_inv[1:, 1:]

        K_m_inv = G - np.outer(f, f) / e # S4

        b = np.array([kernel(tdl(i), x_n) for i in range(n, n + window_size - 1)]).reshape(window_size - 1, 1)
        d = kernel(x_n, x_n) + c

        K_m = np.block([ [A, b], [b.T, d] ]) # S5

        g = 1 / (d - b.T @ K_m_inv @ b)
        f = (- K_m_inv @ b * g).reshape(window_size - 1, 1)
        E = K_m_inv - K_m_inv @ b @ f.T

        K_m_inv = np.block([ [E, f], [f.T, g] ]) # S6

        kernel_values      = K_m[-1]
        kernel_values[-1] -= c

        x_est[n] = kernel_values @ K_m_inv @ y[n : window_size + n] # S7

        # x_est[n] = K_m[-1] @ K_m_inv @ y[n : window_size + n] # S7'

        ## S7 and S7' are similar only if c is small. ##

    return x_est

#----------------------------------------------------------------------------------------------#

def krlsg(x, N, window_size, kernel, d, c = 0.01):

    tdl = lambda x, n: x[n : n + N + 1] # Tapped delay line

    kernel_n = lambda dict_x: np.array([kernel(xn, dict_x[-1]) for xn in dict_x])

    samples = len(d)

    x = np.concatenate((np.zeros(N), x), dtype = x.dtype) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros(samples)

    dict_x = np.array([])
    dict_y = np.array([])

    y = d

    for n in range(samples): # S2

        dict_x = np.concatenate((dict_x, [tdl(x, n)])) if len(dict_x) else np.array([tdl(x, n)])

        dict_y = np.concatenate((dict_y.reshape(len(dict_y), ), [y[n]])) if len(dict_y) else np.array([y[n]])
        # dict_y = np.array(y[max(0, n - window_size + 1) : n + 1])
        dict_y = dict_y.reshape(len(dict_y), 1)

        k = kernel_n(dict_x)

        b = k[: -1].reshape(len(k) - 1, 1)

        d = k[-1] + c

        if b.size > 0:

            g = 1 / (d - b.T @ K_inv @ b)
            f = (-K_inv @ b) * g
            E = K_inv - K_inv @ b @ f.T

            K_inv = np.block([ [E, f], [f.T, g] ])

        else:
            K_inv = np.array([1 / d])

        if len(dict_x) > window_size:

            dict_x = dict_x[1:]
            dict_y = dict_y[1:]

            m = len(K_inv)

            G = K_inv[1 : m, 1:]
            f = K_inv[1 : m, 0].reshape(m - 1, 1)
            e = K_inv[0, 0]

            K_inv = G - f @ f.T / e

            k = k[1:]

        # alpha = K_inv @ dict_y

        x_est[n] = k @ K_inv @ dict_y # k @ alpha

    return x_est

#----------------------------------------------------------------------------------------------#

def krls_ald(x, N, kernel, y, gamma = 0.01):

    samples = len(y)

    x = np.concatenate((np.zeros(N), x), dtype = x.dtype) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    tdl = lambda n: np.flipud(x[n : n + N + 1]) # Tapped delay line (reversed)

    xn = tdl(0)

    K     = kernel(xn, xn).reshape(1, 1)
    K = K if K != 0 else np.array([[1e-6]])
    K_inv = 1 / K
    beta  = y[0] * K_inv

    B = np.ones(1)

    z     = np.zeros((samples, N + 1))
    z[0]  = xn
    z_len = 1

    x_est    = np.zeros(samples)
    x_est[0] = y[0] / (1 + 0.01) # Regularization factor. Just to x_est[0] != y[0].

    for n in range(1, samples):

        xn = tdl(n)

        kernel_list = np.array([kernel(z[i], xn) for i in range(z_len)]).reshape((z_len, 1))

        a = K_inv @ kernel_list

        delta = kernel(xn, xn) - a.T @ kernel_list

        if delta > gamma:

            x_est[n] = kernel_list.T @ beta 

            z[z_len] = xn
            z_len   += 1

            K_inv = 1 / delta * np.block([ [delta * K_inv + np.outer(a, a), -a], [-a.T, 1] ])

            zeros = np.zeros((B.shape[0], 1))

            B = np.block([ [B, zeros], [zeros.T, 1] ])

            tmp = (y[n] - kernel_list.T @ beta) / delta

            beta = np.vstack( (beta - a * tmp, tmp) )

        else:

            tmp_num = B @ a
            tmp_den = 1 + a.T @ tmp_num

            tmp = tmp_num / tmp_den

            B = B - tmp @ a.T @ B

            beta = beta - K_inv @ tmp * (y[n] - kernel_list.T @ beta)

            x_est[n] = kernel_list.T @ beta

    return x_est

#----------------------------------------------------------------------------------------------#

def krls_aldp(x, N, kernel, d, delta = 0.01, lambda_reg = 0.01):

    '''
        Based on the paper.
    '''

    samples = len(d)

    x = np.concatenate((np.zeros(N), x), dtype = x.dtype) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    tdl = lambda n: np.flipud(x[n : n + N + 1]) # Tapped delay line (reversed)

    u_n = tdl(0)

    K_tilde     = kernel(u_n, u_n).reshape(1, 1)
    K_tilde_inv = 1 / K_tilde 
    P           = 1 / (K_tilde + lambda_reg)
    A           = np.ones(1).reshape(1, 1)
    alpha_tilde = P * d[0]

    delta_square = delta ** 2

    C     = np.zeros((samples, N + 1))
    C[0]  = u_n
    C_len = 1

    x_est    = np.zeros(samples)
    x_est[0] = alpha_tilde

    for n in range(1, samples):

        u_n = tdl(n)

        h = np.array([kernel(C[i], u_n) for i in range(C_len)]).reshape((C_len, 1))

        a = K_tilde_inv @ h

        d2 = kernel(u_n, u_n) - h.T @ a

        x_est[n] = h.T @ alpha_tilde

        if d2 <= delta_square:

            s   = a.T @ K_tilde
            tmp = P @ a
            q   = tmp / (1 + s @ tmp)

            alpha_tilde = alpha_tilde - q * (d[n] - s @ alpha_tilde)

            P = P - q @ s @ P

            # A = np.vstack( (A, a.T) ) # Makes no sense. Investigate it!

        else:

            C[C_len] = u_n
            C_len   += 1

            z     = P.T @ h
            zA    = P @ A.T @ A @ h
            tmp_k = kernel(u_n, u_n) + lambda_reg
            gamma = tmp_k - h.T @ zA
            e     = d[n] - h.T @ alpha_tilde
            tmp   = e / gamma

            alpha_tilde = np.vstack( (alpha_tilde - zA * tmp, tmp) )

            P = 1 / gamma * np.block([ [P * gamma + zA @ z.T, -zA], [-z.T, 1] ])

            g = 1 / (tmp_k - h.T @ K_tilde_inv @ h)
            f = (- K_tilde_inv @ h * g).reshape(K_tilde_inv.shape[0], 1)
            E = K_tilde_inv - K_tilde_inv @ h @ f.T

            K_tilde_inv = np.block([ [E, f], [f.T, g] ])

            K_tilde = np.block([ [K_tilde, h], [h.T, tmp_k] ])

            zeros = np.zeros((A.shape[0], 1))

            A = np.block([ [A, zeros], [zeros.T, 1] ])

    return x_est

#----------------------------------------------------------------------------------------------#

def kap(x, N, kernel, d, mu, L, gamma = 0.01, gamma_d = 0.01, gamma_e = 0.01): # 0 < mu <= 1

    x = np.concatenate((np.zeros(N), x), dtype = x.dtype) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    beta = np.array(0, ndmin = 2)

    x_n = np.zeros(N + 1, dtype = x.dtype)

    z = np.zeros((L, N + 1), dtype = x.dtype)

    K     = kernel(x_n, x_n).reshape(1, 1)
    K_inv = 1 / K
    A_inv = K_inv ** 2

    tdl = lambda n: np.flipud(x[n : n + N + 1]) # Tapped delay line (reversed)

    for l in range(1, L):

        x_n = tdl(l)

        z[l] = x_n

        a = kernel(z[0], x_n)
        b = np.array([kernel(z[k], x_n) for k in range(1, l + 1)]).reshape(l, 1)

        K = np.block([ [a, b.T], [b, K[: l, : l]] ])

        A = K.T @ K + gamma * np.eye(l + 1)

        a = A[0, 0]
        b = A[1:, 0].reshape(l, 1)

        C_tilde_inv = A_inv

        C_inv = C_tilde_inv - 1 / (1 + b.T @ C_tilde_inv @ b) * (C_tilde_inv - C_tilde_inv @ b @ b.T @ C_tilde_inv)

        f = a - b.T @ C_inv @ b

        p  = np.array(1, ndmin = 2)
        q  = - C_inv @ b
        qt = q.T
        r  = C_inv * f - C_inv @ b @ qt

        A_inv = 1 / f * np.block([ [p, qt], [q, r] ])

        beta = np.vstack((0, beta))

        e =  np.flipud(d[: l + 1]).reshape(l + 1, 1) - K.T @ beta

        beta = beta + mu * K @ A_inv @ e
        print(l, beta,  mu * K @ A_inv @ e)#, f, C_inv, a, b.T @ C_inv @ b, z[0], x_n)

# #----------------------------------------------------------------------------------------------#
# # MAIN SCRIPT
# #----------------------------------------------------------------------------------------------#

# from kernels import gaussian_kernel

# import matplotlib.pyplot as plt

# def unknown_system(s: np.ndarray, snr = 20):

#     """
#     The linear channel is:

#         H_1(z) = 1 + 0.0668 * z^{-1} - 0.4764 * z^{-2} + 0.8070 * z^{-3}

#     and after 500 symbols it is changed to:

#         H_2(z) = 1 - 0.4326 * z^{-1} - 0.6656 * z^{-2} + 0.7153 * z^{-3}.

#     A binary signal is sent through this channel and then the nonlinear
#     function y = tanh(x) is applied on it, where x is linear channel
#     output. Finally, white Gaussian noise is added to match an SNR of
#     20 dB.

#     Parameters
#     ----------
#     s : np.ndarray
#         Input binary signal.
    
#     Returns
#     -------
#     z : np.ndarray
#         Output signal.
#     """

#     K = len(s)

#     x = np.zeros((K,), dtype = np.float64)

#     s_ext = np.hstack((np.zeros((3,), dtype = np.float64), s))

#     for idx in range(K):

#         if idx < 500:
#             x[idx] = s_ext[idx + 3] + 0.668 * s_ext[idx + 2] - 0.4764 * s_ext[idx + 1] + 0.8070 * s_ext[idx]

#         else:
#             x[idx] = s_ext[idx + 3] - 0.4326 * s_ext[idx + 2] - 0.6656 * s_ext[idx + 1] + 0.7153 * s_ext[idx]

#     y = np.tanh(x)

#     noise = np.random.randn(K)

#     noise *= np.sqrt((np.mean(y ** 2) * 10 ** (- snr / 10)) / np.mean(noise ** 2))

#     return y + noise


# ensemble = 50

# K       = 1500
# N       = 31

# mu_klms = 0.3
# gamma_c = 0.99
# Imax    = 150

# window_size = Imax
# c_krls      = 0.325
# c_krlsg     = 0.325
# c_krls_ald  = 0.01

# mu_kap = 0.3

# # mse_klms     = np.zeros((K, ensemble), dtype = np.float64)
# # mse_krls     = np.zeros((K, ensemble), dtype = np.float64)
# # mse_krlsg    = np.zeros((K, ensemble), dtype = np.float64)
# mse_krls_ald = np.zeros((K, ensemble), dtype = np.float64)
# # mse_kap      = np.zeros((K, ensemble), dtype = np.float64)

# # kernel_app = lambda x, y: gaussian(x, y, 2 * np.ones(N + 1))

# # kernel_klms     = lambda x, y: gaussian_kernel(x, y, np.sqrt(2)  * np.ones(N + 1))
# # kernel_krls     = lambda x, y: gaussian_kernel(x, y, np.sqrt(10) * np.ones(N + 1))
# # kernel_krlsg    = lambda x, y: gaussian_kernel(x, y, np.sqrt(10) * np.ones(N + 1))
# kernel_krls_ald = lambda x, y: gaussian_kernel(x, y, np.sqrt(10) * np.ones(N + 1))
# # kernel_kap      = lambda x, y: gaussian_kernel(x, y, np.sqrt(2)  * np.ones(N + 1))

# for it in range(ensemble):

#     print(it)

#     s = np.random.randint(2, size = (K,)).astype(np.float64)
#     z = unknown_system(s)
    
#     # x_est_klms     = klms(s, N, kernel_klms, z, gamma_c, mu_klms, Imax)
#     # x_est_krls     = krls(s, N, window_size, kernel_krls, z, c_krls)
#     # x_est_krlsg    = krlsg(s, N, window_size, kernel_krlsg, z, c_krlsg)
#     # x_est_krls_ald = krls_ald(s, N, kernel_krls_ald, z, c_krls_ald)
#     x_est_krls_ald = krls_aldp(s, N, kernel_krls_ald, z, c_krls_ald)
#     # x_est_kap      = kap(s, N, kernel_krls, z, mu_kap, Imax)

#     # mse_klms[:, it]     = (z - x_est_klms) ** 2
#     # mse_krls[:, it]     = (z - x_est_krls) ** 2
#     # mse_krlsg[:, it]    = (z - x_est_krlsg) ** 2
#     mse_krls_ald[:, it] = (z - x_est_krls_ald) ** 2
#     # mse_kap[:, it]      = (z - x_est_kap) ** 2

# # mse_klms_avg     = np.mean(mse_klms, axis = 1)
# # mse_krls_avg     = np.mean(mse_krls, axis = 1)
# # mse_krlsg_avg    = np.mean(mse_krlsg, axis = 1)
# mse_krls_ald_avg = np.mean(mse_krls_ald, axis = 1)
# # mse_kap_avg      = np.mean(mse_kap, axis = 1)

# fig = plt.figure()
# ax  = fig.add_subplot(111)

# # ax.plot(10 * np.log10(mse_klms_avg), label = fr'KLMS ($ \mu $ = {mu_klms}, $ \gamma_c $ = {gamma_c}, $ I_{{max}} $ = {Imax})')
# # ax.plot(10 * np.log10(mse_krls_avg), label = fr'KRLS ($ M $ = {window_size}, $ c $ = {c_krls})')
# # ax.plot(10 * np.log10(mse_krlsg_avg), label = fr'KRLSG ($ M $ = {window_size}, $ c $ = {c_krlsg})')
# ax.plot(10 * np.log10(mse_krls_ald_avg), label = fr'KRLS_ALD ($ c $ = {c_krls_ald})')
# # ax.plot(10 * np.log10(mse_kap_avg), label = fr'KAP ($ M $ = {window_size}, $ \mu $ = {mu_kap})')

# ax.grid()
# ax.set_xlabel('Samples -> $ n $')
# ax.set_ylabel('MSE ($ dB $)')

# # fig.tight_layout()

# plt.title(fr'Unknown System (SWKRLS paper example): $ K $ = {K}, $ N $ = {N}, $ \epsilon $ = {ensemble}')

# plt.legend(loc = 'best')

# plt.show()

# #----------------------------------------------------------------------------------------------#
# End of File (EOF)
#----------------------------------------------------------------------------------------------#