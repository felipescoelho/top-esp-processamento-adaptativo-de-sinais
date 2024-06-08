"""e1.py

Script for exercice 1 of the 4th chapter.

luizfelipe.coelho@smt.ufrj.br
Jun 3, 2024
"""


import numpy as np
from ch4.dist_filters.pari_mutuel import pari_mutuel


if __name__ == '__main__':
    rng = np.random.default_rng()
    P = np.array([[.5, .5], [.9, .1]])
    b = np.array([.2, .8])
    pi = rng.random((2,))
    pi /= np.sum(np.abs(pi))

    beta, pi = pari_mutuel(P, b, pi, 100)
    print(beta)
    print(pi)
