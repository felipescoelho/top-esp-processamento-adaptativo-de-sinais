"""chap1.py

Scrip for exercises from Chapter 1 of Online Learning and Adaptive
Filters book.

luizfelipe.coelho@smt.ufrj.br
Mar 17, 2024
"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.adaptive_filters import lms, nlms, ap_alg


def arg_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exercise', type=str, default='2,3,4,5')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parser()
    exercise_list = [int(ex_no) for ex_no in args.exercise.split(',')]
    
    if 2 in exercise_list:
        a = np.array((1, -.8182))
        b = np.array((1, .8182))
        ensemble = 100
        K = 20000
        N = 29
        sigma_n2 = 1e-6
        mse_mc = np.zeros((K, ensemble))
        for it in range(ensemble):
            x = np.random.randn(K,)
            y = lfilter(b, a, x)
            n = np.random.randn(K,)
            n *= np.sqrt(sigma_n2/np.mean(np.abs(n)**2))
            d = y+n
            w0 = np.zeros((N+1,))
            mu = 1/(5*(N+1)*np.mean(np.abs(x)**2))
            _, e, _ = lms(x, d, mu, w0)
            mse_mc[:, it] = e**2
        
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.plot(10*np.log10(np.mean(mse_mc, axis=1)))
        plt.show()

    if 3 in exercise_list:
        a = np.array((1, .9))
        b = np.array((1,))
        ensemble = 100
        K = 20000
        N = 29
        sigma_n2 = 1e-6
        gamma = 1e-12
        mse_lms = np.zeros((K, ensemble))
        mse_nlms = np.zeros((K, ensemble))
        mse_ap = np.zeros((K, ensemble))
        for it in range(ensemble):
            r = np.random.randn(K,)
            x = lfilter(b, a, r)
            n = np.random.randn(K,)
            n *= np.sqrt(sigma_n2/np.mean(np.abs(n)**2))
            d = r+n
            w0 = np.zeros((N+1,))
            mu_lms = 1/(5*(N+1)*np.mean(np.abs(x)**2))
            _, e_lms, _ = lms(x, d, mu_lms, w0)
            mu_nlms = 128/(5*(N+1)*np.mean(np.abs(x)**2))
            _, e_nlms, _ = nlms(x, d, mu_nlms, gamma, w0)
            mu_ap = 16/(5*(N+1)*np.mean(np.abs(x)**2))
            _, e_ap, _ = ap_alg(x, d, mu_ap, gamma, 1, w0)
            mse_lms[:, it] = np.abs(e_lms)**2
            mse_nlms[:, it] = np.abs(e_nlms)**2
            mse_ap[:, it] = np.abs(e_ap[:, 0])**2
        
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.plot(10*np.log10(np.mean(mse_lms, axis=1)),
                 label=f'LMS, $\mu={mu_lms:.4f}$')
        ax0.plot(10*np.log10(np.mean(mse_nlms, axis=1)),
                 label=f'NLMS, $\mu={mu_nlms:.4f}$')
        ax0.plot(10*np.log10(np.mean(mse_ap, axis=1)),
                 label=f'AP, $\mu={mu_ap:.4f}$')
        ax0.legend()
        plt.show()

    if 4 in exercise_list:
        fs = 8*1e3
        T = 1/fs
        Deltat = 20*1e-3
        t = np.arange(0, Deltat, T)
        exc = np.zeros((len(t), 1))
        period = 51
        exc[np.arange(0, len(exc), period)] = 1
        wo = np.array((1.44, -.68, -.42, .24, .37, -.35))
        A = np.hstack(([1], -wo))
        s = lfilter([1], A, exc)
        print(np.nonzero(s-exc))

