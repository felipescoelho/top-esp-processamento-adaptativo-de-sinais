"""main.py

Script for Distributed Algorithms for Array Signal Processing.

luizfelipe.coelho@smt.ufrj.br
May 26, 2024
"""


import numpy as np


if __name__ == '__main__':
    P = 6  # Number of nodes
    Q = 2  # Number of sensors in each node
    adjacency_mat = np.array([[0, 1, 1, 1, 0, 0],
                              [1, 0, 1, 1, 0, 0],
                              [1, 1, 0, 0, 1, 1],
                              [1, 1, 0, 0, 1, 1],
                              [0, 0, 1, 1, 0, 1],
                              [0, 0, 1, 1, 1, 0]], dtype=np.float64)
    degree_mat = np.diag(np.sum(adjacency_mat, axis=0))
    laplacian_mat = degree_mat - adjacency_mat
