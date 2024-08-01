"""main.py

Script for Distributed Algorithms for Array Signal Processing.

luizfelipe.coelho@smt.ufrj.br
May 26, 2024
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import remez
from misc.doa_algorithms import dist_power_method


if __name__ == '__main__':
    rng = np.random.default_rng()
    P = 6  # Number of nodes
    Q = 8  # Number of sensors in each node
    N = int(P*Q)  # Number of sensors
    L = 9  # Filter length
    M = 4  # Decimation ratio
    D = 5  # Number of directions
    J = int(np.floor((N-L+1)/M))
    doa = [np.pi*np.sin(-np.pi/36), np.pi*np.sin(np.pi/36), np.pi/2, .74*np.pi,
           .98*np.pi]  # DOA from simulation parameteres
    A = np.array([[np.exp(1j*doa[d]*n) for d in range(D)] for n in range(N)],
                 dtype=np.complex128)
    ensemble = 2000
    K = 100  # Number of snapshots
    startband = 1/(4*M)
    stopband = 3/(4*M)
    bands = [0, startband, stopband, .5]
    desired = [1, 0]
    h = remez(L, bands, desired)
    adjacency_mat = np.array([[0., 1., 1., 1., 0., 0.],
                              [1., 0., 1., 1., 0., 0.],
                              [1., 1., 0., 0., 1., 1.],
                              [1., 1., 0., 0., 1., 1.],
                              [0., 0., 1., 1., 0., 1.],
                              [0., 0., 1., 1., 1., 0.]], dtype=np.float64)
    degree_mat = np.diag(np.sum(adjacency_mat, axis=0))
    laplacian_mat = degree_mat - adjacency_mat
    cov_mat = np.array([[10**(-.5), 0., 0., 0., 0.],
                        [0., 10**(-.5), 0., 0., 0.],
                        [0., 0., 10**1.5, .6*10**1.5, .6*10**1.5],
                        [0., 0., .6*10**1.5, 10**1.5, .6*10**1.5],
                        [0., 0., .6*10**1.5, .6*10**1.5, 10**1.5]])
    for it in range(ensemble):
        c = rng.multivariate_normal(np.zeros((D,)), cov_mat, (K,)).T
        e = rng.standard_normal((N, K)) + 1j*rng.standard_normal((N, K))
        for k in range(K):
            e[:, k] /= np.sqrt(np.vdot(e[:, k], e[:, k])/N)
        x = A@c + e
    Rxx = x @ np.conj(x.T) / K
    
    print(x.shape)
    print(h.shape)
    print(J)

    # Plot images:
    folderpath = 'array_signal_processing/figs'
    os.makedirs(folderpath, exist_ok=True)
    H = np.fft.fftshift(np.fft.fft(h, 2**10))
    f = np.arange(-1, 1, 1/2**9)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(f, 20*np.log10(np.abs(H)), label='Traditional CBS')
    ymin, ymax = ax.get_ylim()
    ax.vlines(2*startband, ymin, ymax, colors='tab:gray', ls='-.',
              label='Passband edge')
    ax.vlines(-2*startband, ymin, ymax, colors='tab:gray', ls='-.')
    ax.vlines(2*stopband, ymin, ymax, colors='tab:red' ,ls='-.',
              label='Stopband edge')
    ax.vlines(-2*stopband, ymin, ymax, colors='tab:red', ls='-.')
    for idx, true_doa in enumerate(doa):
        if idx == 0:
            ax.vlines(true_doa/np.pi, ymin, ymax, colors='tab:orange',
                      ls='--', label='True DOAs')
        else:
            ax.vlines(true_doa/np.pi, ymin, ymax, colors='tab:orange', ls='--')
    ax.set_xlabel('Normalized Frequency, $\omega/\pi$')
    ax.set_ylabel('Magnitude, dB')
    ax.grid()
    ax.legend()
    ax.set_ylim((ymin, ymax))
    fig.tight_layout()
    figpath = os.path.join(folderpath, 'filter_response.eps')
    fig.savefig(figpath, bbox_inches='tight')

    plt.show()
