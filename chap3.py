"""chap3.py

Script for exercises from Chapter 3 of Online Learning and Adaptive
Filters book.

luizfelipe.coelho@smt.ufrj.br
Mar 28, 2024
"""


import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from src.kernel_based import KLMS


def arg_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exercise', type=str, default='1')
    args = parser.parse_args()

    return args


def unknown_system_easy(x: np.ndarray, sigma_n_2: float):
    """An easy unknown system to teste the algorithms."""
    K = len(x)
    d = np.zeros((K,), dtype=x.dtype)
    x_ext = np.hstack((np.zeros((2,), dtype=x.dtype), x))
    n = np.random.randn(K,) if not np.iscomplex(x).any() \
        else np.random.randn(K,) + 1j*np.random.randn(K,)    
    n *= np.sqrt(sigma_n_2/(np.dot(n, np.conj(n))/K))
    for k in range(2, K+2):
        d[k-2] = -.76*x_ext[k] + 1.0*x_ext[k-1]
    
    return d+n


def unknown_system_3_1(x: np.ndarray, sigma_n_2: float):
    """Unknown system for exercise 3.1"""
    K = len(x)
    d = np.zeros((K,))
    x_ext = np.hstack((np.zeros((2,)), x))
    for k in range(K):
        d[k] = -.76*x_ext[k+2] - 1.0*x_ext[k+1] + 1.0*x_ext[k] \
            + .5 * x_ext[k+2]**2 + 2.0*x_ext[k+2]*x_ext[k] \
            - 1.6 * x_ext[k+1]**2 + 1.2 * x_ext[k]**2 \
            + .8*x_ext[k+1]*x_ext[k]
    n = np.random.randn(K,)
    n *= np.sqrt(sigma_n_2 / np.mean(n**2))

    return d + n


def unkown_system_3_2(x: np.ndarray, sigma_n_2: float):
    """Unknown system for exercise 3.2"""
    K = len(x)
    d = np.zeros((K,), dtype=x.dtype)
    x_ext = np.hstack((np.zeros((2,), dtype=x.dtype), x))
    for k in range(K):
        d[k] = -.08*x_ext[k+2] - .15*x_ext[k+1] + .14*x_ext[k] \
            + .055 * x_ext[k+2]**2 + .3*x_ext[k+2]*x[k] - .16*x[k+1]**2 \
            + .14*x_ext[k]**2
    n = np.random.randn(K,)
    n *= np.sqrt(sigma_n_2 / np.mean(n**2))

    return d + n


if __name__ == '__main__':
    args = arg_parser()
    exercise_list = [int(ex_no) for ex_no in args.exercise.split(',')]

    if 1 in exercise_list:
        # Definitions:
        a = [1, -.95]
        b = [1]
        K = 1500
        Imax = 150
        N = 15
        ensemble = 1
        mu_klms = .04  #.06
        gamma_c = .99
        sigma_x_2 = .1
        sigma_n_2 = 1e-2
        klms_kwargs = {
            'order': N, 'step_factor': mu_klms,
            'kernel_args': (.2*np.ones((N+1,)),),
            'kernel_kwargs': {'kernel_type': 'gauss', 'Imax': Imax,
                              'gamma_c': gamma_c,
                              'data_selection': 'coherence approach'}}
        mse_klms = np.zeros((K, ensemble), dtype=np.float64)
        # Compute:
        for it in range(ensemble):
            x = np.random.randn(K,)
            x *= np.sqrt(sigma_x_2/np.mean(x**2))
            x = lfilter(b, a, x)
            d = unknown_system_3_1(x, sigma_n_2)
            kernel_lms = KLMS(**klms_kwargs)
            _, e = kernel_lms.run_batch(x, d)
            mse_klms[:, it] = e**2
        mse_klms_avg = np.mean(mse_klms, axis=1)
        # Plot:
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.plot(10*np.log10(mse_klms_avg))
        ax0.grid()
        ax0.set_xlabel('Sample, $k$')
        ax0.set_ylabel('MSE, dB')
        fig0.tight_layout()

        plt.show()
