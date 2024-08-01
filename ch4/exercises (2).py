#!/usr/bin/python
# -*- coding: utf-8 -*-


#----------------------------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------------------------#

from distributed import (
                         AtC_LMS,
                         AtC_SM_NLMS,
                         AtC_SM_NLMS_SIC_RFB,
                         CtA_LMS,
                         eisenberg_gale,
                         ILMS,
                         IRLS,
                         np
                        )

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

from scipy.signal import lfilter


#----------------------------------------------------------------------------------------------#
# MAIN CODE
#----------------------------------------------------------------------------------------------#

def example_1(): # 0

    print('Example 1\n')

    P = np.array([ [0.3, 0.7], [0.4, 0.6] ])
    b = np.array([0.2, 0.8])

    return P, b, *eisenberg_gale(P, b)


def exercise_1(): # 1

    print('Exercise 1\n')

    P = np.array([ [0.5, 0.5], [0.9, 0.1] ])
    b = np.array([0.2, 0.8])

    return P, b, *eisenberg_gale(P, b)


def exercise_2(): # 2

    print('Exercise 2\n')

    P = np.array([ [0.3, 0.5, 0.2], [0, 0.4, 0.6] ])
    b = np.array([0.4, 0.6])

    return P, b, *eisenberg_gale(P, b)


def exercise_3(): # 3

    print('Exercise 3\n')

    M = 15
    N = 10

    mu = 0.3
    K  = 1000

    ensemble = 100

    mean     = 0
    variance = 0.01

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_ILMS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_ILMS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_ILMS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_ILMS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_ILMS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_ILMS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    x = np.zeros((M, K))
    d = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):
            x[m] = np.random.randn(K)

        eta = np.random.normal(mean, np.sqrt(variance), K)

        for m in range(M):

            x_tdl = np.concatenate([np.zeros(N), x[m]])

            for k in range(K):
                d[m, k] = np.dot(w_0, np.flipud(x_tdl[k : k + N + 1])) + eta[k]

        x_est_ILMS_4p51, w_m_4p51, e_m_4p51 = ILMS(x, d, N, M, mu, '4.51')
        x_est_ILMS_4p53, w_m_4p53, e_m_4p53 = ILMS(x, d, N, M, mu)

        mse_ILMS_4p51[:, :, it] = (d - x_est_ILMS_4p51) ** 2
        mse_ILMS_4p53[:, :, it] = (d - x_est_ILMS_4p53) ** 2

        for m in range(M):
            for k in range(K):
                w_m_ILMS_4p51[m, k, it] = np.linalg.norm(w_0 - w_m_4p51[m, k])
                w_m_ILMS_4p53[m, k, it] = np.linalg.norm(w_0 - w_m_4p53[m, k])

        e_m_ILMS_4p51[:, :, it] = e_m_4p51 ** 2
        e_m_ILMS_4p53[:, :, it] = e_m_4p53 ** 2

    mse_ILMS_avg_4p51 = np.mean(mse_ILMS_4p51, axis = 2)
    mse_ILMS_avg_4p53 = np.mean(mse_ILMS_4p53, axis = 2)

    w_m_ILMS_avg_4p51 = np.mean(w_m_ILMS_4p51, axis = 2)
    w_m_ILMS_avg_4p53 = np.mean(w_m_ILMS_4p53, axis = 2)

    e_m_ILMS_avg_4p51 = np.mean(e_m_ILMS_4p51, axis = 2)
    e_m_ILMS_avg_4p53 = np.mean(e_m_ILMS_4p53, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_ILMS_avg_4p51[M - 1]), linestyle = 'solid' , label = fr'ILMS (Node {M - 1:02d} - Equation 4.51)')
    ax0.plot(10 * np.log10(mse_ILMS_avg_4p53[M - 1]), linestyle = 'dotted', label = fr'ILMS (Node {M - 1:02d} - Equation 4.53)')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 3: $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_ILMS_avg_4p51[m], linestyle = 'solid' if m < 10 else 'dashed', label = fr'ILMS (Node {m:02d} - Equation 4.51)')
        ax1.plot(w_m_ILMS_avg_4p53[m], linestyle = 'dotted' if m < 10 else 'dashdot', label = fr'ILMS (Node {m:02d} - Equation 4.53)')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 3: $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_ILMS_avg_4p51[M - 1], linestyle = 'solid' , label = fr'ILMS (Node {M - 1:02d} - Equation 4.51)')
    ax2.plot(e_m_ILMS_avg_4p53[M - 1], linestyle = 'dotted', label = fr'ILMS (Node {M - 1:02d} - Equation 4.53)')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 3: $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_4(): # 4

    print('Exercise 4\n')

    M = 15
    N = 10

    mu = 0.3
    K  = 1000

    n_f = 3 # k - 2, k - 1, k

    K_f = K + n_f - 1

    ensemble = 100

    mean     = 0
    variance = 0.01

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_ILMS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_ILMS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_ILMS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_ILMS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_ILMS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_ILMS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    x   = np.zeros((M, K_f))
    x_f = np.zeros((M, K))
    d   = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m] = np.random.randn(K_f)

            for k in range(K):
                x_f[m, k] = (x[m, k + n_f - 1] + 2 * x[m, k - 1 + n_f - 1] + x[m, k - 2 + n_f - 1]) / np.sqrt(6)

            x_f[m] /= np.sqrt(np.mean(np.abs(x_f[m]) ** 2))

        eta = np.random.normal(mean, np.sqrt(variance), K)

        for m in range(M):

            x_tdl = np.concatenate([np.zeros(N), x_f[m]])

            for k in range(K):
                d[m, k] = np.dot(w_0, np.flipud(x_tdl[k : k + N + 1])) + eta[k]

        x_est_4p51, w_m_4p51, e_m_4p51 = ILMS(x_f, d, N, M, mu, '4.51')
        x_est_4p53, w_m_4p53, e_m_4p53 = ILMS(x_f, d, N, M, mu)

        mse_ILMS_4p51[:, :, it] = (d - x_est_4p51) ** 2
        mse_ILMS_4p53[:, :, it] = (d - x_est_4p53) ** 2

        for m in range(M):
            for k in range(K):
                w_m_ILMS_4p51[m, k, it] = np.linalg.norm(w_0 - w_m_4p51[m, k])
                w_m_ILMS_4p53[m, k, it] = np.linalg.norm(w_0 - w_m_4p53[m, k])

        e_m_ILMS_4p51[:, :, it] = e_m_4p51 ** 2
        e_m_ILMS_4p53[:, :, it] = e_m_4p53 ** 2

    mse_ILMS_avg_4p51 = np.mean(mse_ILMS_4p51, axis = 2)
    mse_ILMS_avg_4p53 = np.mean(mse_ILMS_4p53, axis = 2)

    w_m_ILMS_avg_4p51 = np.mean(w_m_ILMS_4p51, axis = 2)
    w_m_ILMS_avg_4p53 = np.mean(w_m_ILMS_4p53, axis = 2)

    e_m_ILMS_avg_4p51 = np.mean(e_m_ILMS_4p51, axis = 2)
    e_m_ILMS_avg_4p53 = np.mean(e_m_ILMS_4p53, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_ILMS_avg_4p51[M - 1]), linestyle = 'solid' , label = fr'ILMS (Node {M - 1:02d} - Equation 4.51)')
    ax0.plot(10 * np.log10(mse_ILMS_avg_4p53[M - 1]), linestyle = 'dotted', label = fr'ILMS (Node {M - 1:02d} - Equation 4.53)')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 4: $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_ILMS_avg_4p51[m], linestyle = 'solid' if m < 10 else 'dashed', label = fr'ILMS (Node {m:02d} - Equation 4.51)')
        ax1.plot(w_m_ILMS_avg_4p53[m], linestyle = 'dotted' if m < 10 else 'dashdot', label = fr'ILMS (Node {m:02d} - Equation 4.53)')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 4: $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_ILMS_avg_4p51[M - 1], linestyle = 'solid' , label = fr'ILMS (Node {M - 1:02d} - Equation 4.51)')
    ax2.plot(e_m_ILMS_avg_4p53[M - 1], linestyle = 'dotted', label = fr'ILMS (Node {M - 1:02d} - Equation 4.53)')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 4: $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_5(): # 5

    print('Exercise 5\n')

    M = 15
    N = 10

    Lambda = 0.9
    K      = 1000

    n_f = 3 # k - 2, k - 1, k

    K_f = K + n_f - 1

    ensemble = 100

    mean     = 0
    variance = 0.01

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_IRLS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_IRLS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_IRLS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_IRLS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_IRLS_4p51 = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_IRLS_4p53 = np.zeros((M, K, ensemble), dtype = np.float64)

    x   = np.zeros((M, K_f))
    x_f = np.zeros((M, K))
    d   = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m] = np.random.randn(K_f)

            for k in range(K):
                x_f[m, k] = (x[m, k + n_f - 1] + 2 * x[m, k - 1 + n_f - 1] + x[m, k - 2 + n_f - 1]) / np.sqrt(6)

            x_f[m] /= np.sqrt(np.mean(np.abs(x_f[m]) ** 2))

        eta = np.random.normal(mean, np.sqrt(variance), K)

        for m in range(M):

            x_tdl = np.concatenate([np.zeros(N), x_f[m]])

            for k in range(K):
                d[m, k] = np.dot(w_0, np.flipud(x_tdl[k : k + N + 1])) + eta[k]

        x_est_4p51, w_m_4p51, e_m_4p51 = IRLS(x_f, d, N, M, Lambda, '4.51')
        x_est_4p53, w_m_4p53, e_m_4p53 = IRLS(x_f, d, N, M, Lambda)

        mse_IRLS_4p51[:, :, it] = (d - x_est_4p51) ** 2
        mse_IRLS_4p53[:, :, it] = (d - x_est_4p53) ** 2

        for m in range(M):
            for k in range(K):
                w_m_IRLS_4p51[m, k, it] = np.linalg.norm(w_0 - w_m_4p51[m, k])
                w_m_IRLS_4p53[m, k, it] = np.linalg.norm(w_0 - w_m_4p53[m, k])

        e_m_IRLS_4p51[:, :, it] = e_m_4p51 ** 2
        e_m_IRLS_4p53[:, :, it] = e_m_4p53 ** 2

    mse_IRLS_avg_4p51 = np.mean(mse_IRLS_4p51, axis = 2)
    mse_IRLS_avg_4p53 = np.mean(mse_IRLS_4p53, axis = 2)

    w_m_IRLS_avg_4p51 = np.mean(w_m_IRLS_4p51, axis = 2)
    w_m_IRLS_avg_4p53 = np.mean(w_m_IRLS_4p53, axis = 2)

    e_m_IRLS_avg_4p51 = np.mean(e_m_IRLS_4p51, axis = 2)
    e_m_IRLS_avg_4p53 = np.mean(e_m_IRLS_4p53, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_IRLS_avg_4p51[M - 1]), linestyle = 'solid' , label = fr'IRLS (Node {M - 1:02d} - Equation 4.51)')
    ax0.plot(10 * np.log10(mse_IRLS_avg_4p53[M - 1]), linestyle = 'dotted', label = fr'IRLS (Node {M - 1:02d} - Equation 4.53)')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 5: $ N $ = {N}, $ \lambda $ = {Lambda}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_IRLS_avg_4p51[m], linestyle = 'solid' if m < 10 else 'dashed', label = fr'IRLS (Node {m:02d} - Equation 4.51)')
        ax1.plot(w_m_IRLS_avg_4p53[m], linestyle = 'dotted' if m < 10 else 'dashdot', label = fr'IRLS (Node {m:02d} - Equation 4.53)')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $ ')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 5: $ N $ = {N}, $ \lambda $ = {Lambda}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_IRLS_avg_4p51[M - 1], linestyle = 'solid' , label = fr'IRLS (Node {M - 1:02d} - Equation 4.51)')
    ax2.plot(e_m_IRLS_avg_4p53[M - 1], linestyle = 'dotted', label = fr'IRLS (Node {M - 1:02d} - Equation 4.53)')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 5: $ N $ = {N}, $ \lambda $ = {Lambda}, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_6(): # 6

    print('Exercise 6\n')

    M = 6
    N = 3

    K   = 1000
    n_f = 3 # k - 2, k - 1, k
    K_f = K + n_f - 1

    mu       = 0.01
    mu_delta = None

    normalized = 0

    ensemble = 100

    N_m = np.array([
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1]
                  ])

    mean     = 0
    variance = 1e-3

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    x   = np.zeros((M, K_f))
    x_f = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m] = np.random.randn(K_f)

            for k in range(K):
                x_f[m, k] = (x[m, k + n_f - 1] + 2 * x[m, k - 1 + n_f - 1] + x[m, k - 2 + n_f - 1]) / np.sqrt(6)

            x_f[m] /= np.sqrt(np.mean(np.abs(x_f[m]) ** 2))

        eta = np.random.normal(mean, np.sqrt(variance), K)

        d = np.array([lfilter(w_0, [1], x_f[m]) + eta for m in range(M)])

        x_est_CtA_LMS, w_m_CtA, e_m_CtA = CtA_LMS(x_f, d, N, N_m, mu, mu_delta, normalized)
        x_est_AtC_LMS, w_m_AtC, e_m_AtC = AtC_LMS(x_f, d, N, N_m, mu, mu_delta, normalized)

        mse_CtA_LMS[:, :, it] = (d - x_est_CtA_LMS) ** 2
        mse_AtC_LMS[:, :, it] = (d - x_est_AtC_LMS) ** 2

        for m in range(M):
            for k in range(K):
                w_m_CtA_LMS[m, k, it] = np.linalg.norm(w_0 - w_m_CtA[m, k])
                w_m_AtC_LMS[m, k, it] = np.linalg.norm(w_0 - w_m_AtC[m, k])

        e_m_CtA_LMS[:, :, it] = e_m_CtA ** 2
        e_m_AtC_LMS[:, :, it] = e_m_AtC ** 2

    mse_CtA_LMS_avg = np.mean(mse_CtA_LMS, axis = 2)
    mse_AtC_LMS_avg = np.mean(mse_AtC_LMS, axis = 2)

    w_m_CtA_LMS_avg = np.mean(w_m_CtA_LMS, axis = 2)
    w_m_AtC_LMS_avg = np.mean(w_m_AtC_LMS, axis = 2)

    e_m_CtA_LMS_avg = np.mean(e_m_CtA_LMS, axis = 2)
    e_m_AtC_LMS_avg = np.mean(e_m_AtC_LMS, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_CtA_LMS_avg[M - 1]), linestyle = 'solid' , label = fr'CtA_LMS (Node {M - 1:02d})')
    ax0.plot(10 * np.log10(mse_AtC_LMS_avg[M - 1]), linestyle = 'dashed', label = fr'AtC_LMS (Node {M - 1:02d})')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_CtA_LMS_avg[m], linestyle = 'solid' , label = fr'CtA_LMS (Node {m:02d})')
        ax1.plot(w_m_AtC_LMS_avg[m], linestyle = 'dashed', label = fr'AtC_LMS (Node {m:02d})')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $ ')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_CtA_LMS_avg[M - 1], linestyle = 'solid' , label = fr'CtA_LMS (Node {M - 1:02d})')
    ax2.plot(e_m_AtC_LMS_avg[M - 1], linestyle = 'dashed', label = fr'AtC_LMS (Node {M - 1:02d})')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_7(): # 7

    print('Exercise 7\n')

    M = 3
    N = 3

    K   = 1000
    n_f = 3 # k - 2, k - 1, k
    K_f = K + n_f - 1

    mu       = 0.01
    mu_delta = 10

    normalized = 0

    ensemble = 100

    N_m = np.ones((M, M))

    mean     = 0
    variance = [1, 1, 1e-3]

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    gamma_m_CtA_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    gamma_m_AtC_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    x   = np.zeros((M, K_f))
    x_f = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m] = np.random.randn(K_f)

            for k in range(K):
                x_f[m, k] = (x[m, k + n_f - 1] + 2 * x[m, k - 1 + n_f - 1] + x[m, k - 2 + n_f - 1]) / np.sqrt(6)

            x_f[m] /= np.sqrt(np.mean(np.abs(x_f[m]) ** 2))

        eta = np.array([np.random.normal(mean, np.sqrt(var), K) for var in variance])

        d = np.array([lfilter(w_0, [1], x_f[m]) + eta[m] for m in range(M)])

        x_est_CtA_LMS, w_m_CtA, e_m_CtA, gamma_m_CtA = CtA_LMS(x_f, d, N, N_m, mu, mu_delta, normalized)
        x_est_AtC_LMS, w_m_AtC, e_m_AtC, gamma_m_AtC = AtC_LMS(x_f, d, N, N_m, mu, mu_delta, normalized)

        mse_CtA_LMS[:, :, it] = (d - x_est_CtA_LMS) ** 2
        mse_AtC_LMS[:, :, it] = (d - x_est_AtC_LMS) ** 2

        for m in range(M):
            for k in range(K):
                w_m_CtA_LMS[m, k, it] = np.linalg.norm(w_0 - w_m_CtA[m, k])
                w_m_AtC_LMS[m, k, it] = np.linalg.norm(w_0 - w_m_AtC[m, k])

        e_m_CtA_LMS[:, :, it] = e_m_CtA ** 2
        e_m_AtC_LMS[:, :, it] = e_m_AtC ** 2

        gamma_m_CtA_LMS[:, :, it] = gamma_m_CtA
        gamma_m_AtC_LMS[:, :, it] = gamma_m_AtC

    mse_CtA_LMS_avg = np.mean(mse_CtA_LMS, axis = 2)
    mse_AtC_LMS_avg = np.mean(mse_AtC_LMS, axis = 2)

    w_m_CtA_LMS_avg = np.mean(w_m_CtA_LMS, axis = 2)
    w_m_AtC_LMS_avg = np.mean(w_m_AtC_LMS, axis = 2)

    e_m_CtA_LMS_avg = np.mean(e_m_CtA_LMS, axis = 2)
    e_m_AtC_LMS_avg = np.mean(e_m_AtC_LMS, axis = 2)

    gamma_m_CtA_LMS_avg = np.mean(gamma_m_CtA_LMS, axis = 2)
    gamma_m_AtC_LMS_avg = np.mean(gamma_m_AtC_LMS, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_CtA_LMS_avg[M - 1]), linestyle = 'solid' , label = fr'CtA_LMS (Node {M - 1:02d})')
    ax0.plot(10 * np.log10(mse_AtC_LMS_avg[M - 1]), linestyle = 'dashed', label = fr'AtC_LMS (Node {M - 1:02d})')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 7: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_CtA_LMS_avg[m], linestyle = 'solid' , label = fr'CtA_LMS (Node {m:02d})')
        ax1.plot(w_m_AtC_LMS_avg[m], linestyle = 'dashed', label = fr'AtC_LMS (Node {m:02d})')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $ ')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 7: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_CtA_LMS_avg[M - 1], linestyle = 'solid' , label = fr'CtA_LMS (Node {M - 1:02d})')
    ax2.plot(e_m_AtC_LMS_avg[M - 1], linestyle = 'dashed', label = fr'AtC_LMS (Node {M - 1:02d})')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 7: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## gamma_m #

    fig3 = plt.figure()

    ax3  = fig3.add_subplot(111)

    for m in range(M):
        ax3.plot(gamma_m_CtA_LMS_avg[m], linestyle = 'solid' , label = fr'CtA_LMS (Node {m:02d})')
        ax3.plot(gamma_m_AtC_LMS_avg[m], linestyle = 'dashed', label = fr'AtC_LMS (Node {m:02d})')

    ax3.grid()

    ax3.set_xlabel('Samples -> $ n $')
    ax3.set_ylabel('$ \gamma_m $')

    ax3.legend(loc = 'best')
    ax3.title.set_text(fr'Exercise 7: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 6: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_8(): # 8

    print('Exercise 8\n')

    M = 6
    N = 3

    K   = 1000
    n_f = 3 # k - 2, k - 1, k
    K_f = K + n_f - 1

    mu       = 0.01
    mu_delta = None

    normalized = 0

    ensemble = 100

    N_m = np.array([
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1]
                  ])

    mean     = 0
    variance = 1e-3

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_AtC_FFF_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_AtC_NFF_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_AtC_FFF_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_AtC_NFF_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_AtC_FFF_LMS = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_AtC_NFF_LMS = np.zeros((M, K, ensemble), dtype = np.float64)

    x   = np.zeros((M, K_f))
    x_f = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m] = np.random.randn(K_f)

            for k in range(K):
                x_f[m, k] = (x[m, k + n_f - 1] + 2 * x[m, k - 1 + n_f - 1] + x[m, k - 2 + n_f - 1]) / np.sqrt(6)

            x_f[m] /= np.sqrt(np.mean(np.abs(x_f[m]) ** 2))

        eta = np.random.normal(mean, np.sqrt(variance), K)

        d = np.array([lfilter(w_0, [1], x_f[m]) + eta for m in range(M)])

        x_est_AtC_FFF_LMS, w_m_AtC_FFF, e_m_AtC_FFF = AtC_LMS(x_f, d, N, N_m, mu, mu_delta, normalized, 1)
        x_est_AtC_NFF_LMS, w_m_AtC_NFF, e_m_AtC_NFF = AtC_LMS(x_f, d, N, N_m, mu, mu_delta, normalized, 0)

        mse_AtC_FFF_LMS[:, :, it] = (d - x_est_AtC_FFF_LMS) ** 2
        mse_AtC_NFF_LMS[:, :, it] = (d - x_est_AtC_NFF_LMS) ** 2

        for m in range(M):
            for k in range(K):
                w_m_AtC_FFF_LMS[m, k, it] = np.linalg.norm(w_0 - w_m_AtC_FFF[m, k])
                w_m_AtC_NFF_LMS[m, k, it] = np.linalg.norm(w_0 - w_m_AtC_NFF[m, k])

        e_m_AtC_FFF_LMS[:, :, it] = e_m_AtC_FFF ** 2
        e_m_AtC_NFF_LMS[:, :, it] = e_m_AtC_NFF ** 2

    mse_AtC_FFF_LMS_avg = np.mean(mse_AtC_FFF_LMS, axis = 2)
    mse_AtC_NFF_LMS_avg = np.mean(mse_AtC_NFF_LMS, axis = 2)

    w_m_AtC_FFF_LMS_avg = np.mean(w_m_AtC_FFF_LMS, axis = 2)
    w_m_AtC_NFF_LMS_avg = np.mean(w_m_AtC_NFF_LMS, axis = 2)

    e_m_AtC_FFF_LMS_avg = np.mean(e_m_AtC_FFF_LMS, axis = 2)
    e_m_AtC_NFF_LMS_avg = np.mean(e_m_AtC_NFF_LMS, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_AtC_FFF_LMS_avg[M - 1]), linestyle = 'solid' , label = fr'AtC_FFF_LMS (Node {M - 1:02d})')
    ax0.plot(10 * np.log10(mse_AtC_NFF_LMS_avg[M - 1]), linestyle = 'dashed', label = fr'AtC_NFF_LMS (Node {M - 1:02d})')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 8: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 8: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_AtC_FFF_LMS_avg[m], linestyle = 'solid' , label = fr'AtC_FFF_LMS (Node {m:02d})')
        ax1.plot(w_m_AtC_NFF_LMS_avg[m], linestyle = 'dashed', label = fr'AtC_NFF_LMS (Node {m:02d})')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $ ')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 8: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 8: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_AtC_FFF_LMS_avg[M - 1], linestyle = 'solid' , label = fr'AtC_FFF_LMS (Node {M - 1:02d})')
    ax2.plot(e_m_AtC_NFF_LMS_avg[M - 1], linestyle = 'dashed', label = fr'AtC_NFF_LMS (Node {M - 1:02d})')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 8 $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ \mu_{{\delta}} = {mu_delta} $, $ K $ = {K}, $ \epsilon $ = {ensemble}' if mu_delta else fr'Exercise 8: $ M $ = {M}, $ N $ = {N}, $ \mu $ = {mu}, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_9(): # 9

    print('Exercise 9\n')

    M = 10
    N = 3

    K = 1000

    ensemble = 100

    N_m = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                  ])

    mean = 0

    sigma_2_m = np.array([0.06, 0.05, 0.06, 0.07, 0.01, 0.03, 0.06, 0.09, 0.03, 0.05])

    gamma_bar = np.sqrt(5 * sigma_2_m)

    a = np.array([1, -0.707])
    b = np.array([1])

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    mse_AtC_SM_NLMS_SIC = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_AtC_SM_NLMS_RFB = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_AtC_SM_NLMS_SIC = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_AtC_SM_NLMS_RFB = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_AtC_SM_NLMS_SIC = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_AtC_SM_NLMS_RFB = np.zeros((M, K, ensemble), dtype = np.float64)

    x = np.zeros((M, K))
    d = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m]  = lfilter(b, a, np.random.randn(K))
            x[m] /= np.sqrt(np.mean(np.abs(x[m]) ** 2))

            d[m] = lfilter(w_0, [1], x[m]) + np.random.normal(mean, np.sqrt(sigma_2_m[m]), K)

        x_est_AtC_SIC, w_m_AtC_SIC, e_m_AtC_SIC = AtC_SM_NLMS_SIC_RFB(x, d, N, N_m, gamma_bar, 0)
        x_est_AtC_RFB, w_m_AtC_RFB, e_m_AtC_RFB = AtC_SM_NLMS_SIC_RFB(x, d, N, N_m, gamma_bar, 1)

        mse_AtC_SM_NLMS_SIC[:, :, it] = (d - x_est_AtC_SIC) ** 2
        mse_AtC_SM_NLMS_RFB[:, :, it] = (d - x_est_AtC_RFB) ** 2

        for m in range(M):
            for k in range(K):
                w_m_AtC_SM_NLMS_SIC[m, k, it] = np.linalg.norm(w_0 - w_m_AtC_SIC[m, k])
                w_m_AtC_SM_NLMS_RFB[m, k, it] = np.linalg.norm(w_0 - w_m_AtC_RFB[m, k])

        e_m_AtC_SM_NLMS_SIC[:, :, it] = e_m_AtC_SIC ** 2
        e_m_AtC_SM_NLMS_RFB[:, :, it] = e_m_AtC_RFB ** 2

    mse_AtC_SIC_SM_NLMS_avg = np.mean(mse_AtC_SM_NLMS_SIC, axis = 2)
    mse_AtC_RFB_SM_NLMS_avg = np.mean(mse_AtC_SM_NLMS_RFB, axis = 2)

    w_m_AtC_SIC_SM_NLMS_avg = np.mean(w_m_AtC_SM_NLMS_SIC, axis = 2)
    w_m_AtC_RFB_SM_NLMS_avg = np.mean(w_m_AtC_SM_NLMS_RFB, axis = 2)

    e_m_AtC_SIC_SM_NLMS_avg = np.mean(e_m_AtC_SM_NLMS_SIC, axis = 2)
    e_m_AtC_RFB_SM_NLMS_avg = np.mean(e_m_AtC_SM_NLMS_RFB, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    ax0.plot(10 * np.log10(mse_AtC_SIC_SM_NLMS_avg[M - 1]), linestyle = 'solid', label = fr'AtC_SIC (Node {M - 1:02d})')
    ax0.plot(10 * np.log10(mse_AtC_RFB_SM_NLMS_avg[M - 1]), linestyle = 'dashed' , label = fr'AtC_RFB (Node {M - 1:02d})')

    ax0.grid()

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.legend(loc = 'best')
    ax0.title.set_text(fr'Exercise 9: $ M $ = {M}, $ N $ = {N}, $ \bar{{\gamma}} = {{\sqrt{{5 \sigma^2_n}}}} $, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_AtC_SIC_SM_NLMS_avg[m], linestyle = 'solid' , label = fr'AtC_SIC (Node {m:02d})')
        ax1.plot(w_m_AtC_RFB_SM_NLMS_avg[m], linestyle = 'dashed', label = fr'AtC_RFB (Node {m:02d})')

    ax1.grid()

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || $ ')

    ax1.legend(loc = 'best')
    ax1.title.set_text(fr'Exercise 9: $ M $ = {M}, $ N $ = {N}, $ \bar{{\gamma}} = {{\sqrt{{5 \sigma^2_n}}}} $, $ K $ = {K}, $ \epsilon $ = {ensemble}')


    ## Quadratic Error #

    fig2 = plt.figure()

    ax2  = fig2.add_subplot(111)

    ax2.plot(e_m_AtC_SIC_SM_NLMS_avg[M - 1], linestyle = 'solid' , label = fr'AtC_SIC (Node {M - 1:02d})')
    ax2.plot(e_m_AtC_RFB_SM_NLMS_avg[M - 1], linestyle = 'dashed', label = fr'AtC_RFB (Node {M - 1:02d})')

    ax2.grid()

    ax2.set_xlabel('Samples -> $ n $')
    ax2.set_ylabel('Quadratic Error')

    ax2.legend(loc = 'best')
    ax2.title.set_text(fr'Exercise 9: $ M $ = {M}, $ N $ = {N}, $ \bar{{\gamma}} = {{\sqrt{{5 \sigma^2_n}}}} $, $ K $ = {K}, $ \epsilon $ = {ensemble}')

    plt.show()


def exercise_11(): # 11

    print('Exercise 11\n')

    '''

        prob_fail = np.maximum(0, (len(nodes_m[m]) - 1) / 10)
        np.random.choice([0, 1], p = [prob_fail, 1 - prob_fail])

    '''

    M = 10
    N = 3

    K = 1000

    ensemble = 100

    N_m = np.ones((M, M))

    mean = 0

    snr = 30

    w_0 = np.ones(N + 1) / np.sqrt(N + 1)

    normalized = 1

    probability_feedforward = 1
    probability_feedback    = 1

    mu       = 0.01
    mu_delta = 10

    gamma_bar = 0.01

    mse_NLMS            = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_SM_NLMS         = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_SM_NLMS_NFF     = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_SM_NLMS_SIC     = np.zeros((M, K, ensemble), dtype = np.float64)
    mse_SM_NLMS_SIC_RFB = np.zeros((M, K, ensemble), dtype = np.float64)

    w_m_NLMS            = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_SM_NLMS         = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_SM_NLMS_NFF     = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_SM_NLMS_SIC     = np.zeros((M, K, ensemble), dtype = np.float64)
    w_m_SM_NLMS_SIC_RFB = np.zeros((M, K, ensemble), dtype = np.float64)

    e_m_NLMS            = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_SM_NLMS         = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_SM_NLMS_NFF     = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_SM_NLMS_SIC     = np.zeros((M, K, ensemble), dtype = np.float64)
    e_m_SM_NLMS_SIC_RFB = np.zeros((M, K, ensemble), dtype = np.float64)


    x = np.zeros((M, K))
    d = np.zeros((M, K))

    for it in range(ensemble):

        print('Iteration', it)

        for m in range(M):

            x[m]  = np.random.randn(K)
            x[m] /= np.sqrt(np.mean(np.abs(x[m]) ** 2))

            d[m] = lfilter(w_0, [1], x[m])

            noise  = np.random.randn(K)
            noise *= np.sqrt((np.mean(d[m] ** 2) * 10 ** (- snr / 10)) / np.mean(noise ** 2))

            x[m] += noise

        x_est_NLMS_prob           , w_m_NLMS_prob           , e_m_NLMS_prob, _         = AtC_LMS(            x, d, N, N_m, mu, mu_delta, normalized, 1, probability_feedforward, probability_feedback)
        x_est_SM_NLMS_prob        , w_m_SM_NLMS_prob        , e_m_SM_NLMS_prob         = AtC_SM_NLMS(        x, d, N, N_m, gamma_bar               , 1, probability_feedforward, probability_feedback)
        x_est_SM_NLMS_NFF_prob    , w_m_SM_NLMS_NFF_prob    , e_m_SM_NLMS_NFF_prob     = AtC_SM_NLMS(        x, d, N, N_m, gamma_bar               , 0, probability_feedforward, probability_feedback)
        x_est_SM_NLMS_SIC_prob    , w_m_SM_NLMS_SIC_prob    , e_m_SM_NLMS_SIC_prob     = AtC_SM_NLMS_SIC_RFB(x, d, N, N_m, gamma_bar               , 0, probability_feedforward, probability_feedback)
        x_est_SM_NLMS_SIC_RFB_prob, w_m_SM_NLMS_SIC_RFB_prob, e_m_SM_NLMS_SIC_RFB_prob = AtC_SM_NLMS_SIC_RFB(x, d, N, N_m, gamma_bar               , 1, probability_feedforward, probability_feedback)

        mse_NLMS[:, :, it]            = (d - x_est_NLMS_prob           ) ** 2
        mse_SM_NLMS[:, :, it]         = (d - x_est_SM_NLMS_prob        ) ** 2
        mse_SM_NLMS_NFF[:, :, it]     = (d - x_est_SM_NLMS_NFF_prob    ) ** 2
        mse_SM_NLMS_SIC[:, :, it]     = (d - x_est_SM_NLMS_SIC_prob    ) ** 2
        mse_SM_NLMS_SIC_RFB[:, :, it] = (d - x_est_SM_NLMS_SIC_RFB_prob) ** 2

        for m in range(M):
            for k in range(K):
                w_m_NLMS[m, k, it]            = np.linalg.norm(w_0 - w_m_NLMS_prob[m, k]           ) ** 2 
                w_m_SM_NLMS[m, k, it]         = np.linalg.norm(w_0 - w_m_SM_NLMS_prob[m, k]        ) ** 2 
                w_m_SM_NLMS_NFF[m, k, it]     = np.linalg.norm(w_0 - w_m_SM_NLMS_NFF_prob[m, k]    ) ** 2 
                w_m_SM_NLMS_SIC[m, k, it]     = np.linalg.norm(w_0 - w_m_SM_NLMS_SIC_prob[m, k]    ) ** 2 
                w_m_SM_NLMS_SIC_RFB[m, k, it] = np.linalg.norm(w_0 - w_m_SM_NLMS_SIC_RFB_prob[m, k]) ** 2 

        e_m_NLMS[:, :, it]            = e_m_NLMS_prob            ** 2
        e_m_SM_NLMS[:, :, it]         = e_m_SM_NLMS_prob         ** 2
        e_m_SM_NLMS_NFF[:, :, it]     = e_m_SM_NLMS_NFF_prob     ** 2
        e_m_SM_NLMS_SIC[:, :, it]     = e_m_SM_NLMS_SIC_prob     ** 2
        e_m_SM_NLMS_SIC_RFB[:, :, it] = e_m_SM_NLMS_SIC_RFB_prob ** 2

    mse_NLMS_avg            = np.mean(mse_NLMS           , axis = 2)
    mse_SM_NLMS_avg         = np.mean(mse_SM_NLMS        , axis = 2)
    mse_SM_NLMS_NFF_avg     = np.mean(mse_SM_NLMS_NFF    , axis = 2)
    mse_SM_NLMS_SIC_avg     = np.mean(mse_SM_NLMS_SIC    , axis = 2)
    mse_SM_NLMS_SIC_RFB_avg = np.mean(mse_SM_NLMS_SIC_RFB, axis = 2)

    w_m_NLMS_avg            = np.mean(w_m_NLMS           , axis = 2)
    w_m_SM_NLMS_avg         = np.mean(w_m_SM_NLMS        , axis = 2)
    w_m_SM_NLMS_NFF_avg     = np.mean(w_m_SM_NLMS_NFF    , axis = 2)
    w_m_SM_NLMS_SIC_avg     = np.mean(w_m_SM_NLMS_SIC    , axis = 2)
    w_m_SM_NLMS_SIC_RFB_avg = np.mean(w_m_SM_NLMS_SIC_RFB, axis = 2)

    e_m_NLMS_avg            = np.mean(e_m_NLMS           , axis = 2)
    e_m_SM_NLMS_avg         = np.mean(e_m_SM_NLMS        , axis = 2)
    e_m_SM_NLMS_NFF_avg     = np.mean(e_m_SM_NLMS_NFF    , axis = 2)
    e_m_SM_NLMS_SIC_avg     = np.mean(e_m_SM_NLMS_SIC    , axis = 2)
    e_m_SM_NLMS_SIC_RFB_avg = np.mean(e_m_SM_NLMS_SIC_RFB, axis = 2)


    ## MSE ##

    fig0 = plt.figure()

    ax0  = fig0.add_subplot(111)

    for m in range(M):
        ax0.plot(10 * np.log10(mse_NLMS_avg[m]           ), 'k')#, label = fr'NLMS (Node {m:02d})')
        ax0.plot(10 * np.log10(mse_SM_NLMS_avg[m]        ), 'b')#, label = fr'SM_NLMS (Node {m:02d})')
        ax0.plot(10 * np.log10(mse_SM_NLMS_NFF_avg[m]    ), 'r')#, label = fr'SM_NLMS_NFF (Node {m:02d})')
        ax0.plot(10 * np.log10(mse_SM_NLMS_SIC_avg[m]    ), 'g')#, label = fr'SM_NLMS_SIC (Node {m:02d})')
        ax0.plot(10 * np.log10(mse_SM_NLMS_SIC_RFB_avg[m]), 'y')#, label = fr'SM_NLMS_SIC_RFB (Node {m:02d})')

    ax0.grid()

    ax0.legend(handles = [Line2D([], [], color = 'k', lw = 3), Line2D([], [], color = 'b', lw = 3), Line2D([], [], color = 'r', lw = 3), Line2D([], [], color = 'g', lw = 3), Line2D([], [], color = 'y', lw = 3)], labels = ['NLMS', 'SM_NLMS', 'SM_NLMS_NFF', 'SM_NLMS_SIC', 'SM_NLMS_SIC_RFB'])

    ax0.set_xlabel('Samples -> $ n $')
    ax0.set_ylabel('MSE ($ dB $)')

    ax0.title.set_text(fr'Exercise 11: All nodes - NLMS X SM-NLMS X SM-NLMS-NFF X SM-NLMS-SIC X SM-NLMS-SIC-RFB - PFF = {probability_feedforward}, PFB = {probability_feedback}, $ \epsilon $ = {ensemble}')


    ## || w_0 - w_m || ##

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    for m in range(M):
        ax1.plot(w_m_NLMS_avg[m]           , 'k')#, label = fr'NLMS (Node {m:02d})')
        ax1.plot(w_m_SM_NLMS_avg[m]        , 'b')#, label = fr'SM_NLMS (Node {m:02d})')
        ax1.plot(w_m_SM_NLMS_NFF_avg[m]    , 'r')#, label = fr'SM_NLMS_NFF (Node {m:02d})')
        ax1.plot(w_m_SM_NLMS_SIC_avg[m]    , 'g')#, label = fr'SM_NLMS_SIC (Node {m:02d})')
        ax1.plot(w_m_SM_NLMS_SIC_RFB_avg[m], 'y')#, label = fr'SM_NLMS_SIC_RFB (Node {m:02d})')

    ax1.grid()

    ax1.legend(handles = [Line2D([], [], color = 'k', lw = 3), Line2D([], [], color = 'b', lw = 3), Line2D([], [], color = 'r', lw = 3), Line2D([], [], color = 'g', lw = 3), Line2D([], [], color = 'y', lw = 3)], labels = ['NLMS', 'SM_NLMS', 'SM_NLMS_NFF', 'SM_NLMS_SIC', 'SM_NLMS_SIC_RFB'])

    ax1.set_xlabel('Samples -> $ n $')
    ax1.set_ylabel(r'$ || w_0 - w_m || ^ 2  $ ')

    ax1.title.set_text(fr'Exercise 11: All nodes - NLMS X SM-NLMS X SM-NLMS-NFF X SM-NLMS-SIC X SM-NLMS-SIC-RFB - PFF = {probability_feedforward}, PFB = {probability_feedback}, $ \epsilon $ = {ensemble}')

    plt.show()


#----------------------------------------------------------------------------------------------#
# MAIN SCRIPT
#----------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    plt.rcParams['font.size'] = 17 # Just to save the figures

    # P, b, pi, beta = example_1()

    exercise_8()


#----------------------------------------------------------------------------------------------#
# End of File (EOF)
#----------------------------------------------------------------------------------------------#