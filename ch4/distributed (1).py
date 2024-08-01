#!/usr/bin/python
# -*- coding: utf-8 -*-


#----------------------------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------------------------#

import numpy as np

from scipy.stats import norm


#----------------------------------------------------------------------------------------------#
# MAIN CODE
#----------------------------------------------------------------------------------------------#

def eisenberg_gale(P, b, K = 100, delta = 0.01):

    M = P.shape[0]
    N = P.shape[1]

    if (b_1_norm := np.sum(b)) != 1:
        b /= b_1_norm

    pi = np.random.uniform(0, 1, N)

    if (pi_1_norm := np.sum(pi)) != 1:
        pi /= pi_1_norm

    beta = np.zeros((M, N))

    for k in range(K): # To evaluate the consensus
        for i in range(M):
            index     = np.argmax(P[i] / pi)
            pi[index] = pi[index] + b[i]

    pi /= np.sum(pi)

    for i in range(M):

        indexes = P[i] / pi

        max = np.max(indexes)

        indexes = np.where(np.logical_and(max - 2 * delta <= indexes, indexes <= max + 2 * delta))[0] # To avoid approximation errors. If not => indexes = np.where(indexes == np.max(indexes))[0]

        beta[i, indexes] = b[i] / len(indexes)

    return pi, beta

#----------------------------------------------------------------------------------------------#

def ILMS(x, d, N, M, mu, equation_number = '4.53'):

    '''

        To calculate the estimated x can be used the equation 4.51 or 4.53.

    '''

    K = x.shape[1]

    w   = np.zeros((K + 1, N + 1), dtype = x.dtype)

    w_m = np.zeros((M + 1, K + 1, N + 1), dtype = x.dtype) # It also has the w_0

    e_m = np.zeros((M, K), dtype = x.dtype)

    x = np.hstack((np.zeros((M, N)), x)) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros((M, K), dtype = x.dtype)

    for k in range(K):

        w_m[0, k + 1] = w[k]

        for m in range(M):

            x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

            x_est[m, k] = np.vdot(w_m[m, k + 1] if equation_number == '4.53' else w[k], x_m)

            e_m[m, k] = d[m, k] - x_est[m, k]
            w_m[m + 1, k + 1] = w_m[m, k + 1] + mu / M * np.conjugate(e_m[m, k]) * x_m

        w[k + 1] = w_m[M, k + 1]

    w_m = w_m[1:, : -1, ] # To remove w_0 and sample k + 1

    return x_est, w_m, e_m

#----------------------------------------------------------------------------------------------#

def IRLS(x, d, N, M, Lambda, equation_number = '4.53'):

    '''

        To calculate the estimated x can be used the equation 4.51 or 4.53.

        0 < Lambda <= 1


    '''

    K = x.shape[1]

    w = np.zeros((K + 1, N + 1), dtype = x.dtype)

    w_m = np.zeros((M + 1, K + 1, N + 1), dtype = x.dtype) # It also has the w_0

    S = np.zeros((K + 1, N + 1, N + 1), dtype = x.dtype) # It also has the S(0)

    for i in range(K + 1):
        S[i] = np.eye(N + 1)

    S_hat_m = np.zeros((M + 1, K + 1, N + 1, N + 1), dtype = x.dtype) # It also has the S_hat_0

    e_m = np.zeros((M, K), dtype = x.dtype)

    x = np.hstack((np.zeros((M, N)), x)) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros((M, K), dtype = x.dtype)

    for k in range(K):

        S_hat_m[0, k] = S[k] / Lambda

        w_m[0, k + 1] = w[k]

        for m in range(M):

            x_m = np.flipud(x[m, k : k + N + 1]).reshape((N + 1, 1)) # Tapped delay line (reversed)

            x_est[m, k] = np.vdot(w_m[m, k + 1] if equation_number == '4.53' else w[k], x_m)

            e_m[m, k] = d[m, k] - x_est[m, k]

            tmp = S_hat_m[m, k] @ x_m
            den = 1 + np.conjugate(x_m.T) @ tmp

            w_m[m + 1, k + 1] = w_m[m, k + 1] + (np.conjugate(e_m[m, k]) * (tmp) / den).reshape(N + 1)

            S_hat_m[m + 1, k] = S_hat_m[m, k] - (tmp @ np.conjugate(x_m.T) @ S_hat_m[m, k]) / den

        S[k + 1] = S_hat_m[M, k]
        w[k + 1] = w_m[M, k + 1]

    w_m = w_m[1:, : -1, ] # To remove w_0 and sample k + 1

    return x_est, w_m, e_m

#----------------------------------------------------------------------------------------------#

def CtA_LMS(x, d, N, N_m, mu, mu_delta = None, normalized = 0):

    '''

        mu, mu_delta > 0

        mu_delta = None => Fair and uniformed information (simpler version)

    '''

    K = x.shape[1]

    M = N_m.shape[0]

    w_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    phi_m = np.zeros((M, K, N + 1), dtype = x.dtype)

    e_m = np.zeros((M, K), dtype = x.dtype)

    x = np.hstack((np.zeros((M, N)), x)) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros((M, K), dtype = x.dtype)

    if mu_delta:

        w_tilde_m = np.zeros((M, K    , N + 1), dtype = x.dtype)

        delta_m = np.zeros((M, K + 1), dtype = x.dtype)

        gamma_m = np.zeros((M, K), dtype = x.dtype)

        nodes_m = [[index for index, value in enumerate(N_m[m]) if value and index != m] for m in range(M)] # To avoid the same computation during the whole iteration

        for k in range(K):

            for m in range(M):

                # Aggregate Step

                x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

                w_tilde_m[m, k] = np.sum(w_m[nodes_m[m], k], axis = 0) / len(nodes_m[m])

                gamma_m[m, k] = 1 / (1 + np.exp(-delta_m[m, k]))

                phi_m[m, k] = gamma_m[m, k] * w_m[m, k] + (1 - gamma_m[m, k]) * w_tilde_m[m, k]

                x_est[m, k] = np.vdot(phi_m[m, k], x_m)


                # Adaptation Step

                e_m[m, k] = d[m, k] - x_est[m, k]

                Nabla = - gamma_m[m, k] * (1 - gamma_m[m, k]) * (np.conjugate(e_m[m, k]) * (np.conjugate(w_m[m, k].T) - np.conjugate(w_tilde_m[m, k].T)) @ x_m + e_m[m, k] * np.vdot(x_m, w_m[m, k] - w_tilde_m[m, k]))

                delta_m[m, k + 1] = delta_m[m, k] - mu_delta * Nabla

                w_m[m, k + 1] = phi_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m / np.vdot(x_m, x_m) if normalized else phi_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m

        w_m = w_m[:, : -1, ] # To remove the sample k + 1

        return x_est, w_m, e_m, gamma_m

    else:

        nodes_m = [[index for index, value in enumerate(N_m[m]) if value] for m in range(M)] # To avoid the same computation during the whole iteration

        for k in range(K):

            for m in range(M):

                # Aggregate Step

                x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

                phi_m[m, k] = np.sum(w_m[nodes_m[m], k], axis = 0) / len(nodes_m[m])

                x_est[m, k] = np.vdot(phi_m[m, k], x_m)


                # Adaptation Step

                e_m[m, k] = d[m, k] - x_est[m, k]

                w_m[m, k + 1] = phi_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m / np.vdot(x_m, x_m) if normalized else phi_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m

        w_m = w_m[:, : -1, ] # To remove the sample k + 1

        return x_est, w_m, e_m

#----------------------------------------------------------------------------------------------#

def AtC_LMS(x, d, N, N_m, mu, mu_delta = None, normalized = 0, FFF = 1, probability_feedforward = 0, probability_feedback = 0):

    '''

        mu, mu_delta > 0

        mu_delta = None => Fair and uniformed information (simpler version)

        FFF = Full Feedforward Traffic
        NFF = No Feedforward Traffic

        FFF = 0 => NFF

    '''

    K = x.shape[1]

    M = N_m.shape[0]

    w_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    phi_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    e_m = np.zeros((M, K), dtype = x.dtype)

    x = np.hstack((np.zeros((M, N)), x)) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros((M, K), dtype = x.dtype)

    if mu_delta:

        delta_m = np.zeros((M, K + 1), dtype = x.dtype)

        gamma_m = np.zeros((M, K), dtype = x.dtype)

        phi_tilde_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

        epsilon_m = np.zeros((M, K), dtype = x.dtype)

        nodes_m = [[index for index, value in enumerate(N_m[m]) if value and index != m] for m in range(M)] # To avoid the same computation during the whole iteration

        for k in range(K):

            # Adaptation Step

            for m in range(M):

                x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

                x_est[m, k] = np.vdot(w_m[m, k], x_m)

                e_m[m, k] = d[m, k] - x_est[m, k]

                if FFF:

                    phi_m[m, k + 1] = w_m[m, k]

                    if probability_feedforward:

                        prob_fail = np.maximum(0, (len(nodes_m[m]) - 1) / 10)

                        nodes_to_share_data = [node for node in nodes_m[m] if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail])]

                    else:
                        nodes_to_share_data = nodes_m[m]

                    for l in nodes_to_share_data:

                        x_l = np.flipud(x[l, k : k + N + 1]) # Tapped delay line (reversed)

                        e_m[l, k] = d[l, k] - np.vdot(phi_m[m, k + 1], x_l)

                        phi_m[m, k + 1] += mu * np.conjugate(e_m[l, k]) * x_l / np.vdot(x_l, x_l) if normalized else mu * np.conjugate(e_m[l, k]) * x_l

                else:
                    phi_m[m, k + 1] = w_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m / np.vdot(x_m, x_m) if normalized else w_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m

                epsilon_m[m, k] = d[m, k] - np.vdot(phi_m[m, k + 1], x_m)


            # Aggregate Step

            for m in range(M):

                if probability_feedback:

                    prob_fail = np.maximum(0, (len(nodes_m[m]) - 1) / 10)

                    nodes_to_share_phi_m_N_m = [node for node in nodes_m[m] if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail])]

                else:
                    nodes_to_share_phi_m_N_m = nodes_m[m]

                if len(nodes_to_share_phi_m_N_m):
                    phi_tilde_m[m, k + 1] = np.sum(phi_m[nodes_to_share_phi_m_N_m, k + 1], axis = 0) / len(nodes_to_share_phi_m_N_m)

                gamma_m[m, k] = 1 / (1 + np.exp(-delta_m[m, k]))

                w_m[m, k + 1] = gamma_m[m, k] * phi_m[m, k + 1] + (1 - gamma_m[m, k]) * phi_tilde_m[m, k + 1]

                Nabla = - gamma_m[m, k] * (1 - gamma_m[m, k]) * (np.conjugate(epsilon_m[m, k]) * (np.conjugate(phi_m[m, k + 1].T) - np.conjugate(phi_tilde_m[m, k + 1].T)) @ x_m + epsilon_m[m, k] * np.vdot(x_m, phi_m[m, k + 1] - phi_tilde_m[m, k + 1]))

                delta_m[m, k + 1] = delta_m[m, k] - mu_delta * Nabla

        w_m = w_m[:, : -1, ] # To remove the sample k + 1

        return x_est, w_m, e_m, gamma_m

    else:

        nodes_m = [[index for index, value in enumerate(N_m[m]) if value] for m in range(M)] # To avoid the same computation during the whole iteration

        for k in range(K):

            # Adaptation Step

            for m in range(M):

                x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

                x_est[m, k] = np.vdot(w_m[m, k], x_m)

                e_m[m, k] = d[m, k] - x_est[m, k]

                if FFF:

                    phi_m[m, k + 1] = w_m[m, k]

                    if probability_feedforward:

                        prob_fail = np.maximum(0, (len(nodes_m[m]) - 1) / 10)

                        nodes_to_share_data = [node for node in nodes_m[m] if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail])]

                    else:
                        nodes_to_share_data = nodes_m[m]

                    for l in nodes_to_share_data:

                        x_l = np.flipud(x[l, k : k + N + 1]) # Tapped delay line (reversed)

                        e_m[l, k] = d[l, k] - np.vdot(phi_m[m, k + 1], x_l)

                        phi_m[m, k + 1] += mu * np.conjugate(e_m[l, k]) * x_l / np.vdot(x_l, x_l) if normalized else mu * np.conjugate(e_m[l, k]) * x_l

                else:
                    phi_m[m, k + 1] = w_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m / np.vdot(x_m, x_m) if normalized else w_m[m, k] + mu * np.conjugate(e_m[m, k]) * x_m


            # Aggregate Step

            for m in range(M):

                if probability_feedback:

                    prob_fail = np.maximum(0, (len(nodes_m[m]) - 1) / 10)

                    nodes_to_share_phi_m_N_m = [node for node in nodes_m[m] if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail])]

                else:
                    nodes_to_share_phi_m_N_m = nodes_m[m]

                if len(nodes_to_share_phi_m_N_m):
                    w_m[m, k + 1] = np.sum(phi_m[nodes_to_share_phi_m_N_m, k + 1], axis = 0) / len(nodes_to_share_phi_m_N_m)

        w_m = w_m[:, : -1, ] # To remove the sample k + 1

        return x_est, w_m, e_m

#----------------------------------------------------------------------------------------------#

def AtC_SM_NLMS(x, d, N, N_m, gamma_bar, FFF = 1, probability_feedforward = 0, probability_feedback = 0):

    '''

        gamma_bar > 0

        FFF = Full Feedforward Traffic
        NFF = No Feedforward Traffic

        FFF = 0 => NFF

    '''

    K = x.shape[1]

    M = N_m.shape[0]

    w_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    phi_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    e_m = np.zeros((M, K), dtype = x.dtype)

    rho_2_m = np.ones((M, K + 1), dtype = x.dtype)

    x = np.hstack((np.zeros((M, N)), x)) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros((M, K), dtype = x.dtype)

    nodes_m = [[index for index, value in enumerate(N_m[m]) if value] for m in range(M)] # To avoid the same computation during the whole iteration

    for k in range(K):

        nodes_to_share_phi_m = set()

        # Adaptation Step

        for m in range(M):

            x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

            x_est[m, k] = np.vdot(w_m[m, k], x_m)

            e_m[m, k] = d[m, k] - x_est[m, k]

            phi_m[m, k + 1] = w_m[m, k]

            if FFF:

                if probability_feedforward:

                    prob_fail = np.maximum(0, (len(nodes_m[m]) - 2) / 10)

                    nodes_to_share_data = [node for node in nodes_m[m] if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail]) or node == m]

                else:
                    nodes_to_share_data = nodes_m[m]

                for l in nodes_to_share_data:

                    x_l = np.flipud(x[l, k : k + N + 1]) # Tapped delay line (reversed)

                    e_m[l, k] = d[l, k] - np.vdot(phi_m[m, k + 1], x_l)

                    if np.abs(e_m[l, k]) > gamma_bar:

                        mu_l = 1 - gamma_bar / np.abs(e_m[l, k])

                        phi_m[m, k + 1] += mu_l * np.conjugate(e_m[l, k]) * x_l / np.vdot(x_l, x_l)

                        # rho_2_m[m, k + 1] -= mu_l ** 2 * e_m[l, k] ** 2 / np.vdot(x_l, x_l)

            elif np.abs(e_m[m, k]) > gamma_bar:

                mu_m = 1 - gamma_bar / np.abs(e_m[m, k])

                phi_m[m, k + 1] += mu_m * np.conjugate(e_m[m, k]) * x_m / np.vdot(x_m, x_m)

                # rho_2_m[m, k + 1] -= mu_m ** 2 * e_m[m, k] ** 2 / np.vdot(x_m, x_m)

            if not np.array_equal(phi_m[m, k + 1], phi_m[m, k]):
                nodes_to_share_phi_m.add(m)


        # Aggregate Step

        for m in range(M):

            if probability_feedback:

                nodes_to_share_phi_m_N_m = []

                node_m = 0

                for node in nodes_m[m]:

                    if node in nodes_to_share_phi_m:

                        nodes_to_share_phi_m_N_m.append(node)

                        if node == m:
                            node_m = 1

                prob_fail = np.maximum(0, (len(nodes_to_share_phi_m_N_m) - 2) / 10) if node_m else np.maximum(0, (len(nodes_to_share_phi_m_N_m) - 1) / 10)

                nodes_to_share_phi_m_N_m = [node for node in nodes_to_share_phi_m_N_m if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail]) or node == m]

            else:
                nodes_to_share_phi_m_N_m = [node for node in nodes_m[m] if node in nodes_to_share_phi_m]

            for l in nodes_to_share_phi_m_N_m:

                tmp = phi_m[m, k + 1] - phi_m[l, k + 1]
                tmp = np.vdot(tmp, tmp)

                if tmp:

                    Lambda = (1 - (rho_2_m[m, k + 1] - rho_2_m[l, k + 1]) / tmp) / 2

                    if Lambda <= 0 or Lambda >= 1:
                        Lambda = 0

                else:
                    Lambda = 0

                phi_m[m, k + 1] = (1 - Lambda) * phi_m[m, k + 1] + Lambda * phi_m[l, k + 1]

                rho_2_m[m, k + 1] = (1 - Lambda) * rho_2_m[m, k + 1] + Lambda * rho_2_m[l, k + 1] - Lambda * (1 - Lambda) * tmp

            w_m[m, k + 1] = phi_m[m, k + 1]

    w_m = w_m[:, : -1, ] # To remove the sample k + 1

    return x_est, w_m, e_m


#----------------------------------------------------------------------------------------------#

def AtC_SM_NLMS_SIC_RFB(x, d, N, N_m, gamma_bar, RFB = 0, probability_feedforward = 0, probability_feedback = 0):

    '''

        gamma_bar > 0

        RFB = Reduced Feedback Traffic

    '''

    K = x.shape[1]

    M = N_m.shape[0]

    if isinstance(gamma_bar, int) or isinstance(gamma_bar, float):
        gamma_bar = gamma_bar * np.ones(M, dtype = x.dtype)

    w_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    phi_m = np.zeros((M, K + 1, N + 1), dtype = x.dtype)

    e_m = np.zeros((M, K), dtype = x.dtype)

    rho_2_m = np.ones((M, K + 1), dtype = x.dtype)

    x = np.hstack((np.zeros((M, N)), x)) # Preparing the input: 0 padding the vector. The tapped delay line will be computed online to save memory

    x_est = np.zeros((M, K), dtype = x.dtype)

    nodes_m = [[index for index, value in enumerate(N_m[m]) if value] for m in range(M)] # To avoid the same computation during the whole iteration

    for k in range(K):

        nodes_to_share_data_Nm = set()
        nodes_to_share_phi_m   = set()

        # Adaptation Step

        for m in range(M):

            x_m = np.flipud(x[m, k : k + N + 1]) # Tapped delay line (reversed)

            x_est[m, k] = np.vdot(w_m[m, k], x_m)

            e_m[m, k] = d[m, k] - x_est[m, k]

            if np.abs(e_m[m, k]) > gamma_bar[m]:
                nodes_to_share_data_Nm.add(m)

            phi_m[m, k + 1] = w_m[m, k]

        for m in nodes_to_share_data_Nm:

            if probability_feedforward:

                prob_fail = np.maximum(0, (len(nodes_m[m]) - 2) / 10)

                nodes_to_share_data = [node for node in nodes_m[m] if np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail]) or node == m]

            else:
                nodes_to_share_data = [node for node in nodes_m[m] if node in nodes_to_share_data_Nm]

            for l in nodes_to_share_data:

                x_l = np.flipud(x[l, k : k + N + 1]) # Tapped delay line (reversed)

                e_m[l, k] = d[l, k] - np.vdot(phi_m[m, k + 1], x_l)

                if np.abs(e_m[l, k]) > gamma_bar[l]:

                    mu_l = 1 - gamma_bar[l] / np.abs(e_m[l, k])

                    phi_m[m, k + 1] += mu_l * np.conjugate(e_m[l, k]) * x_l / np.vdot(x_l, x_l)

                    # rho_2_m[m, k + 1] -= mu_l ** 2 * e_m[l, k] ** 2 / np.vdot(x_l, x_l)

        for m in range(M):

            if RFB:
                if np.abs(e_m[m, k]) > gamma_bar[m]:
                    nodes_to_share_phi_m.add(m)

            else:
                nodes_to_share_phi_m.add(m)


        # Aggregate Step

        for m in range(M):

            if probability_feedback:

                prob_fail = np.maximum(0, (len(nodes_m[m]) - 2) / 10)

                nodes_to_share_phi_m_N_m = [node for node in nodes_m[m] if ((node in nodes_to_share_phi_m and np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail])) or node == m)]

            else:
                nodes_to_share_phi_m_N_m = [node for node in nodes_m[m] if node in nodes_to_share_phi_m]

            for l in nodes_to_share_phi_m_N_m:

                tmp = phi_m[m, k + 1] - phi_m[l, k + 1]
                tmp = np.vdot(tmp, tmp)

                if tmp:

                    Lambda = (1 - (rho_2_m[m, k + 1] - rho_2_m[l, k + 1]) / tmp) / 2

                    if Lambda <= 0 or Lambda >= 1:
                        Lambda = 0

                else:
                    Lambda = 0

                phi_m[m, k + 1] = (1 - Lambda) * phi_m[m, k + 1] + Lambda * phi_m[l, k + 1]

                rho_2_m[m, k + 1] = (1 - Lambda) * rho_2_m[m, k + 1] + Lambda * rho_2_m[l, k + 1] - Lambda * (1 - Lambda) * tmp

            w_m[m, k + 1] = phi_m[m, k + 1]

    w_m = w_m[:, : -1, ] # To remove the sample k + 1

    return x_est, w_m, e_m

#----------------------------------------------------------------------------------------------#

def distributed_detection_LMS(x, N_m, mu, mu_H0, mu_H1, sigma2_H0, sigma2_H1, epsilon, OR_rule = 1, cov_H0 = None, cov_H1 = None):

    '''

        epsilon > 0

        Based on the paper 

        Distributed Cooperative Spectrum Sensing with Adaptive Combining
        (Francisco C. Ribeiro Jr., Marcello L. R. de Campos and Stefan Werner)

        To evaluate the probabilities of false alarm and detection in this paper, the relations obtained in the following
        paper are used:

        Optimal Linear Cooperation for Spectrum Sensing in Cognitive Radio Networks
        (Zhi Quan, Shuguang Cui and Ali H. Sayed)

        However, we will adapt it here, since the model is different.

        Here will be addressed the case where each node is sampled from Gaussians and nothing more (distortion, noise...)


        Complementary cumulative distribution = Survival function

        sf = survival function

        isf = inverse survival function

    '''

    M, K = x.shape

    N_m_index = [np.where(N_m[m] == 1)[0] for m in range(M)]

    if isinstance(mu_H0, int) or isinstance(mu_H0, float):
        mu_H0 = mu_H0 * np.ones(M, dtype = x.dtype)

    if isinstance(mu_H1, int) or isinstance(mu_H1, float):
        mu_H1 = mu_H1 * np.ones(M, dtype = x.dtype)

    if isinstance(sigma2_H0, int) or isinstance(sigma2_H0, float):
        sigma2_H0 = sigma2_H0 * np.ones(M, dtype = x.dtype)

    if isinstance(sigma2_H1, int) or isinstance(sigma2_H1, float):
        sigma2_H1 = sigma2_H1 * np.ones(M, dtype = x.dtype)

    # w = np.zeros((M, K + 1), dtype = x.dtype)
    w = np.ones((M, K + 1), dtype = x.dtype)

    gamma = np.zeros((M, K), dtype = x.dtype)

    e = np.zeros((M, K), dtype = x.dtype)

    H_soft = np.zeros((M, K), dtype = x.dtype)
    H_hard = np.zeros((M, K), dtype = x.dtype) # Final estimative

    prob_false_soft     = np.zeros((M, K), dtype = x.dtype)
    prob_detection_soft = np.zeros((M, K), dtype = x.dtype)

    for k in range(K):

        # Soft Combining

        for m in range(M):

            T = np.dot(w[N_m_index[m], k], x[N_m_index[m], k])

            E_H0 = np.dot(w[N_m_index[m], k], mu_H0[N_m_index[m]])
            E_H1 = np.dot(w[N_m_index[m], k], mu_H1[N_m_index[m]])

            std_dev_H0 = np.sqrt(w[N_m_index[m], k].T @ cov_H0[N_m_index[m]] @ w[N_m_index[m], k]) if cov_H0 else np.sqrt(np.dot(w[N_m_index[m], k] * sigma2_H0[N_m_index[m]], w[N_m_index[m], k])) # Var_H0 ## Check for the correlated case (cov_H0 is not None)
            std_dev_H1 = np.sqrt(w[N_m_index[m], k].T @ cov_H1[N_m_index[m]] @ w[N_m_index[m], k]) if cov_H1 else np.sqrt(np.dot(w[N_m_index[m], k] * sigma2_H1[N_m_index[m]], w[N_m_index[m], k])) # Var_H1 ## Check for the correlated case (cov_H1 is not None)

            gamma[m, k] = E_H0 + norm.isf(epsilon, loc = mu_H0[m], scale = np.sqrt(sigma2_H0[m])) * std_dev_H0

            # print(k, m, T, gamma[m, k], T < gamma[m, k])

            H_soft[m, k] = 0 if T < gamma[m, k] else 1

            prob_false_soft[m, k]     = norm.sf((gamma[m, k] - E_H0) / std_dev_H0)
            prob_detection_soft[m, k] = norm.sf((gamma[m, k] - E_H1) / std_dev_H1)

            w[m, k + 1] = w[m, k]


        # Hard Combining & Adaptive Weight Update

        for m in range(M):

            H_hard[m, k] = 1 if np.sum(H_soft[N_m_index[m], k]) > (0 if OR_rule else len(N_m_index[m]) / 2) else 0

            # H_hard[m, k] = 1 if np.dot(np.log10(np.array([(prob_detection_soft[l, k] / prob_false_soft[l, k]) if H_soft[l, k] else ((1 - prob_detection_soft[l, k]) / (1 - prob_false_soft[l, k])) for l in N_m_index[m]])), H_soft[N_m_index[m], k]) > -1 else 0 # CHECK IT! PAPER ALEMSEGED2019

            e[m, k] = len(N_m_index[m]) * (mu_H1[m] if H_hard[m, k] else mu_H0[m]) - np.dot(w[N_m_index[m], k], x[N_m_index[m], k])

            w[N_m_index[m], k + 1] += 2 * mu * e[m, k] * x[N_m_index[m], k]

    return H_soft, H_hard, prob_false_soft, prob_detection_soft



def distributed_detection_LMS2(x, h, s, v, N_m, mu, sigma2, epsilon, OR_rule = 1, cov_H0 = None, cov_H1 = None):

    '''

        epsilon > 0

        Based on the paper 

        Distributed Cooperative Spectrum Sensing with Adaptive Combining
        (Francisco C. Ribeiro Jr., Marcello L. R. de Campos and Stefan Werner)

        To evaluate the probabilities of false alarm and detection in this paper, the relations obtained in the following
        paper are used:

        Optimal Linear Cooperation for Spectrum Sensing in Cognitive Radio Networks
        (Zhi Quan, Shuguang Cui and Ali H. Sayed)

        Since we do not have a fusion center now, y = u + n becomes y = u. (Equation 3 from reference 9)

        Complementary cumulative distribution = Survival function

        sf = survival function

        isf = inverse survival function

    '''

    M, K = x.shape

    N_m_index = []

    if isinstance(sigma2, int) or isinstance(sigma2, float):
        sigma2 = sigma2 * np.ones(M, dtype = x.dtype)

    w = np.ones((M, K + 1), dtype = x.dtype)

    gamma = np.zeros((M, K), dtype = x.dtype)

    e = np.zeros((M, K), dtype = x.dtype)

    H_soft = np.zeros((M, K), dtype = x.dtype)
    H_hard = np.zeros((M, K), dtype = x.dtype) # Final estimative

    prob_false_soft     = np.zeros((M, K), dtype = x.dtype)
    prob_detection_soft = np.zeros((M, K), dtype = x.dtype)

    prob_false_hard     = np.zeros((M, K), dtype = x.dtype)
    prob_detection_hard = np.zeros((M, K), dtype = x.dtype)

    y = np.zeros(M, dtype = x.dtype) # Energy of the sensors

    Es = np.sum(s ** 2) # Energy of the signal

    g = h ** 2

    mu_H0 = []
    mu_H1 = []

    Sigma2_H0 = []
    Sigma2_H1 = []

    for m in range(M):

        N_m_index.append(np.where(N_m[m] == 1)[0])

        y[m] = np.sum(x[m] ** 2)

        mu_H0.append(K * sigma2[N_m_index[m]])
        mu_H1.append(K * sigma2[N_m_index[m]] + Es * g[N_m_index[m]])

        Sigma2_H0.append(2 * K * sigma2[N_m_index[m]] ** 2)
        Sigma2_H1.append(2 * K * sigma2[N_m_index[m]] ** 2 + 4 * Es * g[N_m_index[m]] * sigma2[N_m_index[m]])

    mu_H0 = np.array(mu_H0, dtype = object)
    mu_H1 = np.array(mu_H1, dtype = object)

    Sigma2_H0 = np.array(Sigma2_H0, dtype = object)
    Sigma2_H1 = np.array(Sigma2_H1, dtype = object)
    print()
    for k in range(K):

        # Soft Combining

        for m in range(M):

            T = np.dot(w[N_m_index[m], k], y[N_m_index[m]])

            E_H0 = np.dot(w[N_m_index[m], k], mu_H0[m])
            E_H1 = np.dot(w[N_m_index[m], k], mu_H1[m])

            std_dev_H0 = np.sqrt(np.dot(w[N_m_index[m], k] * Sigma2_H0[m], w[N_m_index[m], k])) # Var_H0 ## Check for the correlated case (cov_H0 is not None)
            std_dev_H1 = np.sqrt(np.dot(w[N_m_index[m], k] * Sigma2_H1[m], w[N_m_index[m], k])) # Var_H1 ## Check for the correlated case (cov_H1 is not None)

            gamma[m, k] = E_H0 + norm.isf(epsilon) * std_dev_H0

            # print(k, m, T, gamma[m, k], T < gamma[m, k])

            H_soft[m, k] = 0 if T < gamma[m, k] else 1

            prob_false_soft[m, k]     = norm.sf((gamma[m, k] - E_H0) / std_dev_H0)
            prob_detection_soft[m, k] = norm.sf((gamma[m, k] - E_H1) / std_dev_H1)

            w[m, k + 1] = w[m, k]


        # Hard Combining & Adaptive Weight Update

        for m in range(M):

            H_hard[m, k] = 1 if np.sum(H_soft[N_m_index[m], k]) > (0 if OR_rule else len(N_m_index[m]) / 2) else 0

            # H_hard[m, k] = 1 if np.dot(np.log10(np.array([(prob_detection_soft[l, k] / prob_false_soft[l, k]) if H_soft[l, k] else ((1 - prob_detection_soft[l, k]) / (1 - prob_false_soft[l, k])) for l in N_m_index[m]])), H_soft[N_m_index[m], k]) > -1 else 0 # CHECK IT! PAPER ALEMSEGED2019

            e[m, k] = np.sum(mu_H1[m] if H_hard[m, k] else mu_H0[m]) - np.dot(w[N_m_index[m], k], y[N_m_index[m]])
            print(e[m, k], e[m, k] * y[N_m_index[m]] * y[N_m_index[m]], np.dot(w[N_m_index[m], k], y[N_m_index[m]]))
            w[N_m_index[m], k + 1] += 2 * mu * e[m, k] * y[N_m_index[m]]

    return H_soft, H_hard, prob_false_soft, prob_detection_soft, prob_false_hard, prob_detection_hard

#----------------------------------------------------------------------------------------------#
# MAIN SCRIPT
#----------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    N_m = np.array([
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                   ])

    M = N_m.shape[0]

    N_m_index = [np.where(N_m[m] == 1)[0] for m in range(M)]

    K = 500#100000

    mu = 0.01

    amount_epsilon = 40

    epsilon = np.linspace(0.01, 0.95, amount_epsilon)

    mu_H0 = 0
    mu_H1 = 2

    sigma2_H0 = sigma2_H1 = 1

    x = np.zeros((M, K))

    r  = np.zeros((M, K))
    rc = np.zeros((M, K))

    '''

        Model:
               H_0: x_i(k) = v_i(k)             , i = 1, 2, ..., M
               H_1: x_i(k) = h_i * s(k) + v_i(k), i = 1, 2, ..., M

    '''

    h = np.ones(M)
    s = np.zeros(K)
    v = np.zeros((M, K))

    H0 = np.zeros((M, amount_epsilon), dtype = int)
    H1 = np.zeros((M, amount_epsilon), dtype = int)

    H0_original_samples = np.zeros((M, amount_epsilon), dtype = int)
    H1_original_samples = np.zeros((M, amount_epsilon), dtype = int)

    pd_freq_node   = np.zeros((M, amount_epsilon))
    pd_freq_global = np.zeros(amount_epsilon)

    pd_freq_H_node = np.zeros((M, amount_epsilon))
    pf_freq_H_node = np.zeros((M, amount_epsilon))

    pd_freq_H_global = np.zeros(amount_epsilon)
    pf_freq_H_global = np.zeros(amount_epsilon)

    pds_final = np.zeros((M, amount_epsilon))

    # f = 1 / (K * N_m.shape[0])

    print('Monte-Carlo = 1')

    for it in range(amount_epsilon):

        f_ind = 0

        for k in range(K):

            s[k] = np.random.normal(loc = mu_H1, scale = np.sqrt(sigma2_H1))

            for m in range(M):

                pf = 0.5#np.random.random() # 0.5 if H0 and H1 are equiprobable

                r[m, k] = np.random.choice([0, 1], p = [pf, 1 - pf])

                if r[m, k]:
                    f_ind += 1

                v[m, k] = np.random.normal(loc = mu_H0, scale = np.sqrt(sigma2_H0))

                x[m, k] = (h[m] * s[k] + v[m, k]) if r[m, k] else v[m, k] 

            for m in range(M):
                rc[m, k] = 1 if np.sum(r[N_m_index[m], k]) else 0

        print(f'\n\nepsilon = {epsilon[it]} and p_f = {pf}\n')

        hs, hh, pfs, pds = distributed_detection_LMS(x, N_m, mu, mu_H0, mu_H1, sigma2_H0, sigma2_H1, epsilon[it])
        # hs, hh, pfs, pds = distributed_detection_LMS2(x, h, s, v, N_m, mu, sigma2_H0, epsilon[it])

        for m in range(M):

            for k in range(K):

                if r[m, k]:

                    H1_original_samples[m, it] += 1

                    if hs[m, k]:
                        H1[m, it] += 1

                else:

                    H0_original_samples[m, it] += 1

                    if not hs[m, k]:
                        H0[m, it] += 1

        pd_freq_node[:, it] = (H0[:, it] + H1[:, it]) / (H0_original_samples[:, it] + H1_original_samples[:, it])
        pd_freq_global[it]  = np.sum(H0[:, it] + H1[:, it]) / np.sum(H0_original_samples[:, it] + H1_original_samples[:, it])

        pd_freq_H_node[:, it] = H1[:, it] / H1_original_samples[:, it]
        pf_freq_H_node[:, it] = H0[:, it] / H0_original_samples[:, it]

        pd_freq_H_global[it] = np.sum(H1[:, it]) / np.sum(H1_original_samples[:, it])
        pf_freq_H_global[it] = np.sum(H0[:, it]) / np.sum(H0_original_samples[:, it])

        for m in range(M):
            pds_final[m, it] = np.median(pds[m])
            print('Failure:', 1 - pds_final[m, it])

        print('\nGlobal Detection:', pd_freq_global[it])


    fig0, ax0 = plt.subplots()

    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow', 'cyan', 'lime', 'indigo', 'black']

    for m in range(M):
        ax0.plot(epsilon[0], 1 - pds_final[m, 0], marker = 'o', color = colors[m], label = f'Node {m}')
        for i in range(1, amount_epsilon):
            ax0.plot(epsilon[i], 1 - pds_final[m, i], marker = 'o', color = colors[m])

    ax0.legend()

    ax0.set_xlabel('$ P_f $')
    ax0.set_ylabel(r'$ 1 - P_d $')

    ax0.set_yscale('log')

    ax0.set_xlim(0, 1)
    # ax0.set_ylim(1e-9, 1)

    # ax0.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])


    fig1, ax1 = plt.subplots()

    for m in range(1):
        ax1.plot(pf_freq_H_node[m, 0], 1 - pd_freq_H_node[m, 0], marker = 'o', color = colors[m], label = f'Node {m}')
        # print(m, 0, pf_freq_H_node[m, 0], 1 - pd_freq_H_node[m, 0])
        for i in range(1, amount_epsilon):
            # print(m, i, pf_freq_H_node[m, i], 1 - pd_freq_H_node[m, i])
            ax1.plot(pf_freq_H_node[m, i], 1 - pd_freq_H_node[m, i], marker = 'o', color = colors[m])

    ax1.legend()

    ax1.set_xlabel('$ P_f $')
    ax1.set_ylabel(r'$ 1 - P_d $')

    ax1.set_yscale('log')

    ax1.set_xlim(0, 1)
    # ax1.set_ylim(1e-9, 1)

    # ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])


    plt.show()

#----------------------------------------------------------------------------------------------#
# End of File (EOF)
#----------------------------------------------------------------------------------------------#