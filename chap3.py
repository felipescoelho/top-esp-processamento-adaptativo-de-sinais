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
from src.kernel_based import k_lms


def arg_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--exercise', type=str, default='1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parser()
    exercise_list = [int(ex_no) for ex_no in args.exercise.split(',')]

    if 1 in exercise_list:
        def unknown_system(x):
            K = len(x)
            d = np.zeros((K,), dtype=x.dtype)
            x = np.hstack((np.zeros((2,), dtype=x.dtype), x))
            for k in range(K):
                d[k] = -.76*x[k] - 1.0*x[k-1] + 1.0*x[k-2] + .5*x[k]**2 \
                    + 2.0*x[k]*x[k-2] - 1.6*x[k-1]**2 + 1.2*x[k-2]**2 \
                    +.8*x[k-1]*x[k-2]
            return d
        a = [1, -.95]
        b = [1]
        K = 2000
        sigma_x_2 = .1
        sigma_n_2 = 1e-2
        x = np.random.randn(K,)
        x *= np.sqrt(sigma_x_2/np.mean(np.abs(x)**2))
        x = lfilter(b, a, x)
        n = np.random.randn(K,)
        n *= np.sqrt(sigma_x_2/np.mean(np.abs(n)**2))
        d = unknown_system(x) + n
        arguments = (.2,)
        e = k_lms(x, d, .16, 10, .1, 15, 'gaussian', arguments)
        print(e)
