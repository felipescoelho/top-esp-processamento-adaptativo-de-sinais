"""e2.py

Script for exercice 2 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 4, 2024
"""


import numpy as np
from ch4.dist_filters.pari_mutuel import pari_mutuel


if __name__ == '__main__':
    rng = np.random.default_rng()
    P = np.array([[.3, .5, .2], [0, .4, .6]])
    M, N = P.shape
    b = np.array([.4, .6])
    pi = rng.random((N,))
    pi /= np.sum(np.abs(pi))

    beta, pi = pari_mutuel(P, b, pi, 100)
    print(beta)
    print(pi)
