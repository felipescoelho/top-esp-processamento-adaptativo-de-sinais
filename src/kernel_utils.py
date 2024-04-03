"""kernel_utils.py

Kernel functions necessary to implement the kernel-based adaptive
filters.

luizfelipe.coelho@smt.ufrj.br
Mar 28, 2024
"""


from abc import ABC, abstractmethod
import numpy as np
from numba import stencil, njit


def cosine_similarity(x0: np.ndarray, x1: np.ndarray):
    """Cosine Similarity Kernel.

    Computes: (x0.T @ x1)/(||x0||_2 ||x1||_2)
    
    Parameters
    ----------
    x0 : np.ndarray
        First array in the comparison.
    x1 : np.ndarray
        Second array in the comparison.
    
    Returns
    -------
    y : float
        Resulting similarity between x0 and x1.
    """

    y = np.dot(x0, x1)/(np.linalg.norm(x0, 2)*np.linalg.norm(x1, 2))

    return y


def sigmoid(x0: np.ndarray, x1: np.ndarray, a: float, b: float):
    """Sigmoid Kernel

    Computes: tanh(a * x0.T @ x1 + b)
    
    Parameters
    ----------
    x0 : np.ndarray
        First array in the comparison.
    x1 : np.ndarray
        Second array in the comparison.
    a : float
        Real constant.
    b : float
        Real constant.
    
    Returns
    -------
    y : float
        Resulting similarity between x0 and x1.
    """

    y = np.tanh(a*np.dot(x0, x1) + b)

    return y


def polynomial(x0: np.ndarray, x1: np.ndarray, a: float, b: float, n: int):
    """Polynomial Kernel

    Computes: (a * x0.T @ x1 + b)**n

    Parameters
    ----------
    x0 : np.ndarray
        First array in the comparison.
    x1 : np.ndarray
        Second array in the comparison.
    a : float
        Real constant.
    b : float
        Real constant. b >= 0 is required to guarantee that the kernel
        matrix is positive definite.
    n : int
        Defines the order (degree) of the polynomial.
    
    Returns
    -------
    y : float
        Resulting similarity between x0 and x1.
    """

    y = (a*np.dot(x0, x1) + b)**n

    return y


def gaussian(x0: np.ndarray, x1: np.ndarray, sigma: np.ndarray):
    """Gaussian Kernel
    
    Computes: e**(-.5 * (x0-x1).T @ diag(sigma)**(-1) @ (x0-x1))

    Parameters
    ----------
    x0 : np.ndarray
        First array in comparison.
    x1 : np.ndarray
        Second array in comparison
    sigma : np.ndarray
        Array containing variance for the exponential funcion.

    Returns
    -------
    y : np.ndarray
        Resulting similarity between x0 and x1
    """

    I = len(x0)-1
    if type(sigma) == type(.1):
        sigma = sigma*np.ones((I+1,))
    y = np.exp(-.5* (x0-x1).reshape(1, I+1) @ np.linalg.inv(np.diagflat(sigma))
               @ (x0-x1).reshape(I+1, 1))
    
    return y


def laplacian(x0: np.ndarray, x1: np.ndarray, sigma: float):
    """Laplacian Kernel
    
    Computes e**(-||x0-x1||/sigma)
    
    Parameters
    ----------
    x0 : np.ndarray
        First array in comparison.
    x1 : np.ndarray
        Second array in comparison
    sigma : float
        Bandwidth of the kernel.

    Returns
    -------
    y : np.ndarray
        Resulting similarity between x0 and x1
    """

    y = np.exp(-np.linalg.norm(x0-x1, 1)/sigma)

    return y


class KernelHandlerBase(ABC):
    """Base class to handle the kernel for adaptive filters.
    
    Attributes
    ----------
    name : str
        Name of the kernel function used.
    kfun : function
        Kernel function.
    """

    def __init__(self, *args, **kwargs):
        kernel_dict = {'cosine': cosine_similarity, 'sig': sigmoid,
                       'poly': polynomial, 'gauss': gaussian,
                       'laplace': laplacian}
        self.order = kwargs['order']
        self.name = kwargs['kernel_type']
        self.data_selection = kwargs['data_selection']
        self.kfun = kernel_dict[self.name]
        self.kernel_args = args

    @abstractmethod
    def compute(self):
        """"""

    @abstractmethod
    def update(self):
        """"""


class KernelHandlerLMS(KernelHandlerBase):
    """Class to handle the kernel function for the KMLS algorithm.
    
    Attributes
    ----------
    name : str
        Name of the kernel fuction used.
    kfun : function
        Kernel function.
    kdict_data : np.ndarray
        Array containing all the data for the kernel dictionary.
    kdict_idx : np.ndarray
        Array containing all the indexes for the kernel dicitionary.
    Imax : int
        Maximum number of entries for the kernel dictionary.
    """

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self.Icount = 1  # Initail number of elements in dictionary.
        match self.data_selection:
            case 'coherence approach':
                self.Imax = kwargs['Imax']
                self.gamma_c = kwargs['gamma_c']
                self.kdict_input = np.zeros((self.order+1, self.Imax))
                self.kdict_error = np.zeros((self.Imax,))
                self.lmax = 0
            case 'no selection':
                self.kdict_input = np.zeros((self.order+1, 1))
                self.kdict_error = np.zeros((1,))

    def compute(self, xk: np.ndarray, step_size: float):
        """"""
        match self.data_selection:
            case 'coherence approach':
                I = min(self.Imax, self.Icount)
            case 'no selection':
                I = self.Icount
        kernel_values = np.zeros((I,))
        for i in range(I):
            kernel_values[i] = self.kfun(self.kdict_input[:, i], xk,
                                         *self.kernel_args)
        gk = 2*step_size*np.sum(kernel_values*self.kdict_error[:I]) \
            + 2*step_size*self.kfun(xk, xk, *self.kernel_args)

        return kernel_values, gk[0]

    def update(self, kernel_values: np.ndarray, xk: np.ndarray, ek: float,
               step_size: float):
        """"""
        match self.data_selection:
            case 'coherence approach':
                if np.max(np.abs(kernel_values)) <= self.gamma_c:
                    self.lmax = np.argmax(np.abs(kernel_values))
                    self.Icount += 1
                    if self.Icount <= self.Imax:
                        self.kdict_input[:, self.Icount-1] = xk
                        self.kdict_error[self.Icount-1] = ek
                    else:
                        self.kdict_input[:, self.lmax] = xk
                        self.kdict_error[self.lmax] = ek
                else:
                    self.kdict_error[self.lmax] += step_size*ek
            case 'no selection':
                self.kdict_input = np.hstack((xk.reshape(self.order+1, 1),
                                              self.kdict_input))
                self.kdict_error = np.hstack((ek, self.kdict_error))
                self.Icount += 1


class KernelHandlerAP(KernelHandlerBase):
    """"""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self.Icount = 0
        self.L = kwargs['L']
        match self.data_selection:
            case 'coherence approach':
                self.Imax = kwargs['Imax']
                self.gamma_c = kwargs['gamma_c']
                self.kdict_input = np.zeros((self.order+1, self.L+1, self.Imax))
                self.kernel_mat = np.zeros(())
        

    def compute(self):
        return super().compute()
    
    def update(self):
        return super().update()