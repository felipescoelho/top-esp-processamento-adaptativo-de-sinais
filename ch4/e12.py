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
    K = int(1e5)
    N = 4
    M = 12
    mu = 1e-5
    ensemble = 1
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
    A += np.eye(M)
    for it in range(ensemble):
        
        X = rng.standard_normal()

    