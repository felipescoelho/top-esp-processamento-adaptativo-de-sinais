"""e12.py

Script for exercise 12 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 6, 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from dist_filters.distributed_detection import dist_detection_lms


if __name__ == '__main__':
    rng = np.random.default_rng()
    K = 200
    N = 4
    M = 12
    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    P = A + np.eye(M)
    P = np.diag(np.array([1/Nm for Nm in np.sum(P, axis=0)])) @ P
    