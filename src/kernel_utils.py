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


def gaussian(x0: np.ndarray, x1: np.ndarray, sigma_2: np.ndarray):
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

    dif = x0-x1
    y = np.exp(-.5*np.dot(dif, dif/sigma_2))

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
        if self.name == 'gauss':
            assert args[0].ndim == 1, 'sigma should be a 1d array.'

    @abstractmethod
    def compute(self):
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
            case 'novelty criterion':
                self.Imax = kwargs['Imax']
                self.gamma_d = kwargs['gamma_d']
                self.gamma_e = kwargs['gamma_e']
                self.kdict_input = np.zeros((self.order+1, self.Imax))
                self.kdict_error = np.zeros((self.Imax,))
                self.lmin = 0

    def compute(self, xk: np.ndarray, step_size: float):
        """"""
        match self.data_selection:
            case 'coherence approach' | 'novelty criterion':
                I = min(self.Imax, self.Icount)
            case 'no selection':
                I = self.Icount
        kernel_values = np.zeros((I,))
        for i in range(I):
            kernel_values[i] = self.kfun(self.kdict_input[:, i], xk,
                                         *self.kernel_args)
        gk = 2*step_size*np.sum(kernel_values*self.kdict_error[:I])# \
            # + 2*step_size*self.kfun(xk, xk, *self.kernel_args)

        return kernel_values, gk

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
            case 'novelty criterion':
                I = min(self.Imax, self.Icount)
                values = np.linalg.norm(np.outer(xk, np.ones((I,)))
                                        - self.kdict_input[:, :I],
                                        axis=0)**2
                if np.min(values) >= self.gamma_d \
                        and np.abs(ek) > self.gamma_e:
                    self.lmin = np.argmin(values)
                    self.Icount += 1
                    if self.Icount <= self.Imax:
                        self.kdict_input[:, self.Icount-1] = xk
                        self.kdict_error[self.Icount-1] = ek
                    else:
                        self.kdict_input[:, self.lmin] = xk
                        self.kdict_error[self.lmin] = ek
                else:
                    self.kdict_error[self.lmin] += step_size*ek


class KernelHandlerAP(KernelHandlerBase):
    """"""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self.L = kwargs['L']
        self.gamma = kwargs['gamma']
        match self.data_selection:
            case 'coherence approach':
                self.k = 0
                self.Icount = 1
                self.Imax = kwargs['Imax']
                self.gamma_c = kwargs['gamma_c']
                self.input_AP = np.zeros((self.order+1, self.L+1))
                self.input_dict = np.zeros((self.order+1, self.Imax))
                self.kernel_mat = np.ones((self.Imax, self.L+1))
                self.proj_AP_inv = np.ones((self.L+1, self.L+1))
                self.weight_vector = np.array((0.,))
                self.update_flag = kwargs['dict_update']

    def compute(self, xk: np.ndarray, d_AP: np.ndarray, step_size: float):
        self.k += 1
        self.input_AP = np.column_stack((xk, self.input_AP[:, :-1]))
        match self.data_selection:
            case 'coherence approach':
                if self.k <= self.L:
                    self.input_dict = np.column_stack(
                        (xk, self.input_dict[:, :-1])
                    )
                    kernel_at_k = np.zeros((self.k,))
                    kernel_at_AP = np.zeros((self.k,))
                    for l in range(1, self.k+1):
                        kernel_at_k[l-1] = self.kfun(self.input_dict[:, l], xk,
                                                     *self.kernel_args)
                        kernel_at_AP[l-1] = self.kfun(self.input_dict[:, 0],
                                                      self.input_AP[:, l],
                                                      *self.kernel_args)
                    self.kernel_mat[:self.k+1, :self.k+1] = np.row_stack((
                        np.hstack((self.kfun(self.input_dict[:, 0], xk,
                                             *self.kernel_args), kernel_at_AP)),
                        np.column_stack((kernel_at_k,
                                         self.kernel_mat[:self.k, :self.k]))
                    ))
                    # AP matrix:
                    proj_AP = self.kernel_mat[:self.k+1, :self.k+1].T \
                        @ self.kernel_mat[:self.k+1, :self.k+1] \
                        + self.gamma*np.eye(self.k+1)
                    # Generate the inverse of AP matrix: 
                    C_tilde_inv = self.proj_AP_inv[:self.k, :self.k]
                    a = proj_AP[0, 0]
                    b = proj_AP[1:, 0]
                    C_inv = C_tilde_inv - (1/(1 + b@C_tilde_inv@b)) \
                        * (C_tilde_inv - C_tilde_inv@np.outer(b, b)@C_tilde_inv)
                    f = a - b.T @ C_inv @ b
                    C_invb = C_inv @ b
                    self.proj_AP_inv[:self.k+1, :self.k+1] = (1/f) * np.row_stack((
                        np.hstack(([1], -C_invb)),
                        np.column_stack((
                            -C_invb, C_inv*(f*np.eye(self.k)+np.outer(b, C_invb))
                        ))
                    ))
                    self.weight_vector = np.hstack(([0], self.weight_vector))
                    y_AP = np.dot(self.kernel_mat[:self.k+1, :self.k+1].T,
                                  self.weight_vector)
                    y_AP = np.hstack((y_AP, np.zeros((self.L-self.k,))))
                    e_AP = d_AP - y_AP
                    self.weight_vector += step_size \
                        * self.kernel_mat[:self.Icount+1, :]@self.proj_AP_inv@e_AP
                    self.Icount += 1
                if self.k >= self.L+1:
                    if self.Icount < self.Imax:
                        kernel_dict = np.zeros((self.Icount,))
                        for l in range(self.Icount):
                            kernel_dict[l] = self.kfun(
                                self.input_dict[:, l], xk, *self.kernel_args
                            )
                        if np.max(np.abs(kernel_dict)) <= self.gamma_c:
                            self.input_dict = np.column_stack(
                                (xk, self.input_dict[:, :-1])
                            )
                            kernel_at_k = np.zeros((self.Icount,))
                            kernel_at_AP = np.zeros((self.L,))
                            for l in range(1, self.Icount+1):
                                kernel_at_k[l-1] = self.kfun(
                                    self.input_dict[:, l], xk, *self.kernel_args
                                )
                            for l in range(1, self.L+1):
                                kernel_at_AP[l-1] = self.kfun(
                                    self.input_dict[:, 0], self.input_AP[:, l],
                                    *self.kernel_args
                                )
                            self.kernel_mat[:self.Icount+1, :] = \
                                np.row_stack((
                                    np.hstack((self.kfun(self.input_dict[:, 0],
                                                         xk, *self.kernel_args),
                                               kernel_at_AP
                                    )), np.column_stack((
                                        kernel_at_k,
                                        self.kernel_mat[:self.Icount, :self.L]
                                    ))
                                ))
                            self.weight_vector = np.hstack(([0], self.weight_vector))
                            y_AP = np.dot(self.kernel_mat[:self.Icount+1, :].T,
                                          self.weight_vector)
                            e_AP = d_AP - y_AP
                            proj_AP = self.kernel_mat[:self.Icount+1, :].T \
                                @ self.kernel_mat[:self.Icount+1, :] \
                                + self.gamma*np.eye(self.L+1)
                            A_hat = self.proj_AP_inv[:-1, :-1]
                            b_hat = self.proj_AP_inv[:-1, -1]
                            c_hat = self.proj_AP_inv[-1, -1] \
                                if self.proj_AP_inv[-1, -1] != 0 else self.gamma
                            C_tilde_inv = A_hat - np.outer(b_hat, b_hat)/c_hat
                            a = proj_AP[0, 0]
                            b = proj_AP[1:, 0]
                            C_inv = C_tilde_inv - (1/(1 + b@C_tilde_inv@b)) \
                                * (C_tilde_inv - C_tilde_inv@np.outer(b, b)@C_tilde_inv)
                            f = a - b.T @ C_inv @ b
                            C_invb = C_inv @ b
                            self.proj_AP_inv = (1/f) * np.row_stack((
                                np.hstack(([1], -C_invb)),
                                np.column_stack((
                                    -C_invb, C_inv*(f*np.eye(self.L)
                                                    + np.outer(b, C_invb))
                                ))
                            ))
                            self.weight_vector += step_size \
                                * self.kernel_mat[:self.Icount+1, :] \
                                @ self.proj_AP_inv@e_AP
                            self.Icount += 1
                        else:
                            y_AP = np.dot(self.kernel_mat[:self.Icount, :].T,
                                  self.weight_vector)
                            e_AP = d_AP - y_AP
                            self.weight_vector += step_size \
                                * self.kernel_mat[:self.Icount, :] \
                                @ self.proj_AP_inv @ e_AP
                    else:
                        if self.update_flag:
                            kernel_dict = np.zeros((self.Imax,))
                            for l in range(self.Imax):
                                kernel_dict[l] = self.kfun(
                                    self.input_dict[:, l], xk,
                                    *self.kernel_args
                                )
                            if np.max(np.abs(kernel_dict)) <= self.gamma_c:
                                self.input_dict = np.column_stack(
                                    (xk, self.input_dict[:, :-1])
                                )
                                kernel_at_k = np.zeros((self.Imax-1,))
                                kernel_at_AP = np.zeros((self.L,))
                                for l in range(1, self.Imax):
                                    kernel_at_k[l-1] = self.kfun(
                                        self.input_dict[:, l], xk,
                                        *self.kernel_args
                                    )
                                for l in range(1, self.L+1):
                                    kernel_at_AP[l-1] = self.kfun(
                                        self.input_dict[:, 0],
                                        self.input_AP[:, l], *self.kernel_args
                                    )
                                self.kernel_mat = np.row_stack((
                                    np.hstack((self.kfun(self.input_dict[:, 0],
                                                        xk, *self.kernel_args),
                                            kernel_at_AP
                                    )), np.column_stack((
                                        kernel_at_k,
                                        self.kernel_mat[:self.Imax-1, :self.L]
                                    ))
                                ))
                                self.weight_vector = np.hstack(([0], self.weight_vector[:-1]))
                                y_AP = np.dot(self.kernel_mat.T, self.weight_vector)
                                e_AP = d_AP - y_AP
                                proj_AP = self.kernel_mat.T @ self.kernel_mat \
                                    + self.gamma*np.eye(self.L+1)
                                A_hat = self.proj_AP_inv[:-1, :-1]
                                b_hat = self.proj_AP_inv[:-1, -1]
                                c_hat = self.proj_AP_inv[-1, -1] \
                                    if self.proj_AP_inv[-1, -1] != 0 else self.gamma
                                C_tilde_inv = A_hat - np.outer(b_hat, b_hat)/c_hat
                                a = proj_AP[0, 0]
                                b = proj_AP[1:, 0]
                                C_inv = C_tilde_inv - (1/(1 + b@C_tilde_inv@b)) \
                                    * (C_tilde_inv - C_tilde_inv@np.outer(b, b)@C_tilde_inv)
                                f = a - b.T @ C_inv @ b
                                C_invb = C_inv @ b
                                self.proj_AP_inv = (1/f) * np.row_stack((
                                    np.hstack(([1], -C_invb)),
                                    np.column_stack((
                                        -C_invb, C_inv*(f*np.eye(self.L)
                                                        + np.outer(b, C_invb))
                                    ))
                                ))
                                self.weight_vector += step_size \
                                    * self.kernel_mat \
                                    @ self.proj_AP_inv@e_AP
                            else:
                                y_AP = np.dot(self.kernel_mat.T, self.weight_vector)
                                e_AP = d_AP - y_AP
                                self.weight_vector += step_size \
                                    * self.kernel_mat@self.proj_AP_inv@e_AP
                        else:
                            y_AP = np.dot(self.kernel_mat.T, self.weight_vector)
                            e_AP = d_AP - y_AP
                            self.weight_vector += step_size \
                                * self.kernel_mat@self.proj_AP_inv@e_AP

        return y_AP, e_AP


class KernelHandlerSMAP(KernelHandlerBase):
    """"""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        self.L = kwargs['L']
        self.gamma = kwargs['gamma']
        match self.data_selection:
            case 'coherence approach':
                self.k = 0
                self.Icount = 1
                self.Imax = kwargs['Imax']
                self.gamma_c = kwargs['gamma_c']
                self.input_AP = np.zeros((self.order+1, self.L+1))
                self.input_dict = np.zeros((self.order+1, self.Imax))
                self.kernel_mat = np.zeros((self.Imax, self.L+1))
                self.proj_AP_inv = np.ones((self.L+1, self.L+1))
                self.weight_vector = np.array((0.,))
                self.update_flag = kwargs['dict_update']

    def __update_inv_proj(self, I: int):
        r"""Method to update the inverted matrix of afine projection.
        .. {\bf A}_{\textrm{AP}}^{-1}
        """
        if I >= self.L:
            proj_AP = (self.kernel_mat[:I, :].T @ self.kernel_mat[:I, :]
                       + self.gamma*np.eye(self.L+1))
            A_hat = self.proj_AP_inv[:-1, :-1]
            b_hat = self.proj_AP_inv[-1, :-1]
            c_hat = self.proj_AP_inv[-1, -1] + 1e-12
            C_tild_inv = A_hat - np.outer(b_hat, b_hat)/c_hat
            L = self.L
        else:
            proj_AP = (self.kernel_mat[:I, :I+1].T @ self.kernel_mat[:I, :I+1]
                       + self.gamma*np.eye(I+1))
            C_tild_inv = self.proj_AP_inv[:I, :I]
            L = I
        a = proj_AP[0, 0]
        b = proj_AP[1:, 0]
        C_inv = C_tild_inv - (1/(1 + b@C_tild_inv@b)) \
            * (C_tild_inv - C_tild_inv@np.outer(b, b)@C_tild_inv)
        f = a - b.T @ C_inv @ b
        C_invb = C_inv @ b
        self.proj_AP_inv[:L+1, :L+1] = (1/f) * np.hstack((
            np.atleast_2d(np.hstack(([1], -C_invb))).T,
            np.vstack((-C_invb, C_inv*(f*np.eye(L)+np.outer(b, C_invb))))
        ))

    def __update_kernel_mat(self, kernel_AP: np.ndarray, I: int):
        """Method to update the kernel matrix."""
        L = min(I, self.L)
        top_l = self.kfun(self.input_dict[:, 0], self.input_AP[:, 0],
                          *self.kernel_args)
        top_r = np.zeros((L,))
        for l in range(L):
            top_r[l] = self.kfun(self.input_dict[:, 0], self.input_AP[:, l+1],
                                 *self.kernel_args)
        top_v = np.hstack((top_l, top_r))
        b_2d = np.atleast_2d(kernel_AP)  # b as a 2D array
        lower_m = np.hstack((b_2d.T, self.kernel_mat[:I, :L]))
        self.kernel_mat[:I+1, :L+1] = np.vstack((top_v, lower_m))
    
    def compute(self, xk: np.ndarray, d_AP: np.ndarray, step_size: float,
                gamma_bar: float):
        self.k += 1
        self.input_AP = np.column_stack((xk, self.input_AP[:, :-1]))
        match self.data_selection:
            case 'coherence approach':
                if self.k <= self.L:
                    self.input_dict = np.column_stack(
                        (xk, self.input_dict[:, :-1])
                    )
                    kernel_at_k = np.zeros((self.k,))
                    kernel_at_AP = np.zeros((self.k,))
                    for l in range(1, self.k+1):
                        kernel_at_k[l-1] = self.kfun(self.input_dict[:, l], xk,
                                                     *self.kernel_args)
                        kernel_at_AP[l-1] = self.kfun(self.input_dict[:, 0],
                                                      self.input_AP[:, l],
                                                      *self.kernel_args)
                    self.kernel_mat[:self.k+1, :self.k+1] = np.row_stack((
                        np.hstack((self.kfun(self.input_dict[:, 0], xk,
                                             *self.kernel_args), kernel_at_AP)),
                        np.column_stack((kernel_at_k,
                                         self.kernel_mat[:self.k, :self.k]))
                    ))
                    # AP matrix:
                    proj_AP = self.kernel_mat[:self.k+1, :self.k+1].T \
                        @ self.kernel_mat[:self.k+1, :self.k+1] \
                        + self.gamma*np.eye(self.k+1)
                    # Generate the inverse of AP matrix: 
                    C_tilde_inv = self.proj_AP_inv[:self.k, :self.k]
                    a = proj_AP[0, 0]
                    b = proj_AP[1:, 0]
                    C_inv = C_tilde_inv - (1/(1 + b@C_tilde_inv@b)) \
                        * (C_tilde_inv - C_tilde_inv@np.outer(b, b)@C_tilde_inv)
                    f = a - b.T @ C_inv @ b
                    C_invb = C_inv @ b
                    self.proj_AP_inv[:self.k+1, :self.k+1] = (1/f) * np.row_stack((
                        np.hstack(([1], -C_invb)),
                        np.column_stack((
                            -C_invb, C_inv*(f*np.eye(self.k)+np.outer(b, C_invb))
                        ))
                    ))
                    self.weight_vector = np.hstack(([0], self.weight_vector))
                    y_AP = np.dot(self.kernel_mat[:self.k+1, :self.k+1].T,
                                  self.weight_vector)
                    y_AP = np.hstack((y_AP, np.zeros((self.L-self.k,))))
                    e_AP = d_AP - y_AP
                    self.weight_vector += step_size \
                        * self.kernel_mat[:self.Icount+1, :]@self.proj_AP_inv@e_AP
                    self.Icount += 1
                if self.k >= self.L+1:
                    if self.Icount < self.Imax:
                        kernel_dict = np.zeros((self.Icount,))
                        for l in range(self.Icount):
                            kernel_dict[l] = self.kfun(
                                self.input_dict[:, l], xk, *self.kernel_args
                            )
                        if np.max(np.abs(kernel_dict)) <= self.gamma_c:
                            self.input_dict = np.column_stack(
                                (xk, self.input_dict[:, :-1])
                            )
                            kernel_at_k = np.zeros((self.Icount,))
                            kernel_at_AP = np.zeros((self.L,))
                            for l in range(1, self.Icount+1):
                                kernel_at_k[l-1] = self.kfun(
                                    self.input_dict[:, l], xk, *self.kernel_args
                                )
                            for l in range(1, self.L+1):
                                kernel_at_AP[l-1] = self.kfun(
                                    self.input_dict[:, 0], self.input_AP[:, l],
                                    *self.kernel_args
                                )
                            self.kernel_mat[:self.Icount+1, :] = np.row_stack((
                                np.hstack((self.kfun(self.input_dict[:, 0], xk,
                                                     *self.kernel_args),
                                           kernel_at_AP
                                )), np.column_stack((
                                    kernel_at_k,
                                    self.kernel_mat[:self.Icount, :self.L]
                                ))
                            ))
                            self.weight_vector = np.hstack(([0], self.weight_vector))
                            y_AP = np.dot(self.kernel_mat[:self.Icount+1, :].T,
                                          self.weight_vector)
                            e_AP = d_AP - y_AP
                            proj_AP = self.kernel_mat[:self.Icount+1, :].T \
                                @ self.kernel_mat[:self.Icount+1, :] \
                                + self.gamma*np.eye(self.L+1)
                            A_hat = self.proj_AP_inv[:-1, :-1]
                            b_hat = self.proj_AP_inv[:-1, -1]
                            c_hat = self.proj_AP_inv[-1, -1] \
                                if self.proj_AP_inv[-1, -1] != 0 else self.gamma
                            C_tilde_inv = A_hat - np.outer(b_hat, b_hat)/c_hat
                            a = proj_AP[0, 0]
                            b = proj_AP[1:, 0]
                            C_inv = C_tilde_inv - (1/(1 + b@C_tilde_inv@b)) \
                                * (C_tilde_inv - C_tilde_inv@np.outer(b, b)@C_tilde_inv)
                            f = a - b.T @ C_inv @ b
                            C_invb = C_inv @ b
                            self.proj_AP_inv = (1/f) * np.row_stack((
                                np.hstack(([1], -C_invb)),
                                np.column_stack((
                                    -C_invb, C_inv*(f*np.eye(self.L)
                                                    + np.outer(b, C_invb))
                                ))
                            ))
                            # SM:
                            step = 1 - gamma_bar/np.abs(e_AP[0]) \
                                if np.abs(e_AP[0]) > gamma_bar else 0
                            self.weight_vector += step*e_AP[0] \
                                * self.kernel_mat[:self.Icount+1, :] \
                                @ self.proj_AP_inv \
                                @ np.hstack(([1], np.zeros((self.L,))))
                            self.Icount += 1
                        else:
                            y_AP = np.dot(self.kernel_mat[:self.Icount, :].T,
                                  self.weight_vector)
                            e_AP = d_AP - y_AP
                            # SM:
                            step = 1 - gamma_bar/np.abs(e_AP[0]) \
                                if np.abs(e_AP[0]) > gamma_bar else 0
                            self.weight_vector += step*e_AP[0] \
                                * self.kernel_mat[:self.Icount, :] \
                                @ self.proj_AP_inv \
                                @ np.hstack(([1], np.zeros((self.L,))))
                    else:
                        if self.update_flag:
                            kernel_dict = np.zeros((self.Imax,))
                            for l in range(self.Imax):
                                kernel_dict[l] = self.kfun(
                                    self.input_dict[:, l], xk,
                                    *self.kernel_args
                                )
                            if np.max(np.abs(kernel_dict)) <= self.gamma_c:
                                self.input_dict = np.column_stack(
                                    (xk, self.input_dict[:, :-1])
                                )
                                kernel_at_k = np.zeros((self.Imax-1,))
                                kernel_at_AP = np.zeros((self.L,))
                                for l in range(1, self.Imax):
                                    kernel_at_k[l-1] = self.kfun(
                                        self.input_dict[:, l], xk,
                                        *self.kernel_args
                                    )
                                for l in range(1, self.L+1):
                                    kernel_at_AP[l-1] = self.kfun(
                                        self.input_dict[:, 0],
                                        self.input_AP[:, l], *self.kernel_args
                                    )
                                self.kernel_mat = np.row_stack((
                                    np.hstack((self.kfun(self.input_dict[:, 0],
                                                        xk, *self.kernel_args),
                                            kernel_at_AP
                                    )), np.column_stack((
                                        kernel_at_k,
                                        self.kernel_mat[:self.Imax-1, :self.L]
                                    ))
                                ))
                                self.weight_vector = np.hstack(([0], self.weight_vector[:-1]))
                                y_AP = np.dot(self.kernel_mat.T, self.weight_vector)
                                e_AP = d_AP - y_AP
                                proj_AP = self.kernel_mat.T @ self.kernel_mat \
                                    + self.gamma*np.eye(self.L+1)
                                A_hat = self.proj_AP_inv[:-1, :-1]
                                b_hat = self.proj_AP_inv[:-1, -1]
                                c_hat = self.proj_AP_inv[-1, -1] \
                                    if self.proj_AP_inv[-1, -1] != 0 else self.gamma
                                C_tilde_inv = A_hat - np.outer(b_hat, b_hat)/c_hat
                                a = proj_AP[0, 0]
                                b = proj_AP[1:, 0]
                                C_inv = C_tilde_inv - (1/(1 + b@C_tilde_inv@b)) \
                                    * (C_tilde_inv - C_tilde_inv@np.outer(b, b)@C_tilde_inv)
                                f = a - b.T @ C_inv @ b
                                C_invb = C_inv @ b
                                self.proj_AP_inv = (1/f) * np.row_stack((
                                    np.hstack(([1], -C_invb)),
                                    np.column_stack((
                                        -C_invb, C_inv*(f*np.eye(self.L)
                                                        + np.outer(b, C_invb))
                                    ))
                                ))
                                # SM:
                                step = 1 - gamma_bar/np.abs(e_AP[0]) \
                                    if np.abs(e_AP[0]) > gamma_bar else 0
                                self.weight_vector += step*e_AP[0] \
                                    * self.kernel_mat@self.proj_AP_inv \
                                    @ np.hstack(([1], np.zeros((self.L,))))
                            else:
                                y_AP = np.dot(self.kernel_mat.T, self.weight_vector)
                                e_AP = d_AP - y_AP
                                # SM:
                                step = 1 - gamma_bar/np.abs(e_AP[0]) \
                                    if np.abs(e_AP[0]) > gamma_bar else 0
                                self.weight_vector += step*e_AP[0] \
                                    * self.kernel_mat@self.proj_AP_inv \
                                    @ np.hstack(([1], np.zeros((self.L,))))
                        else:
                            y_AP = np.dot(self.kernel_mat.T, self.weight_vector)
                            e_AP = d_AP - y_AP
                            # SM:
                            step = 1 - gamma_bar/np.abs(e_AP[0])\
                                if np.abs(e_AP[0]) > gamma_bar else 0
                            self.weight_vector += step_size*e_AP[0] \
                                * self.kernel_mat@self.proj_AP_inv \
                                @ np.hstack(([1], np.zeros((self.L,))))

        return y_AP, e_AP
    

class KernelHandlerRLS(KernelHandlerBase):
    """"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, xk):
        return super().compute()
    
    def update(self):
        return super().update()