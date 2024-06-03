"""kernel_based.py

Script with kernel-based adaptive filter algorithms.

luizfelipe.coelho@smt.ufrj.br
Mar 28, 2024
"""


import numpy as np
from abc import ABC, abstractmethod
from .kernel_utils import KernelHandlerLMS, KernelHandlerAP, KernelHandlerSMAP


class KernelAdaptiveFiltersBase(ABC):
    """An abstract class for all kernel-based adaptive filters."""

    def __init__(self, **kwargs):
        """"""
        self.order = kwargs['order']
        self.kfun_args = kwargs['kernel_args']
        self.kfun_kwargs = kwargs['kernel_kwargs']

    @abstractmethod
    def evaluate(self):
        """Method to run Kernel Adaptive Filter.
        
        Parameters
        ----------
        xk : np.ndarray
            Input signal as 1d array containing N+1 samples
        dk : float
            Sample of the desired signal
        
        Returns
        -------
        gk : np.ndarray
            Sample of the output signal
        ek : np.ndarray
            Sample of the error signal
        """

    @abstractmethod
    def run_batch(self, x: np.ndarray, d: np.ndarray):
        """Method to run Kernel Adaptive Filter.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal as 1d array
        d : np.ndarray
            Desired signal as 1d array
        
        Returns
        -------
        g : np.ndarray
            Output signal as 1d array
        e : np.ndarray
            Error signal as 1d array
        """

        K = len(d)
        if len(x) == len(d):
            x = np.hstack((np.zeros((self.order,), dtype=x.dtype), x))
        elif len(x) != len(d) + self.order:
            raise ValueError('dimension mismatch.')
        g = np.zeros((K,), dtype=x.dtype)
        e = np.zeros((K,), dtype=x.dtype)
        for k in range(K):
            tdl = np.flipud(x[k:self.order+k+1])  # tapped delay line
            g[k], e[k] = self.evaluate(tdl, d[k])

        return g, e


class KLMS(KernelAdaptiveFiltersBase):
    """Class for Kernel Least Mean Squares (KLMS) Adaptive Filter.
    
    Attributes
    ----------
    order : int
        The order of the filter. In the case of kernel-based filters the
        order only defines the length of the tapped delay line, or the
        number of past samples we consider in each iteration.
    mu : float
        Step factor. mu < 2, ussually smaller than 1.
    Imax: int
        Number of samples in the kernel dicitionary.
    kfun: function
        Kernel function. The selected kernel function.
    kfun_args: tuple
        Set of arguments for the kernel function.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Set of keyword arguments that defines the adaptive filter.
            
            Valid keyword arguments are:

            kernel_type : str
                
            order : int
                Defines the number of past samples used in each
                iteration.
            step_factor : float
                Defines the step size factor used in to update the
                filter.
            kernel_args : tuple
                Arguments for the kernel function.
            kernel_kwargs : dict
                Set of keyword arguments that defines the kernel
                function and how it is handled.

                Valid keyword arguments are:

                Imax : int
                    Maximum number of elements in the kernel function.
                gamma_c : float
                    Coherence threshold between new incoming data and
                    previously stored data. This is used for the data
                    selection process.
                kernel_type : str
                    Defines the kernel function used (see code for more
                    information on possible kernel functions).
                data_selection : str
                    Defines a data selection heuristic for exploring
                    data sparsity in the kernel domain (see code for
                    more information on possible data selection
                    heuristics).
        """

        super().__init__(**kwargs)
        self.mu = kwargs['step_factor']
        self.kfun_kwargs['order'] = self.order
        self.kernel = KernelHandlerLMS(*self.kfun_args, **self.kfun_kwargs)
    
    def evaluate(self, xk: np.ndarray, dk: float):
        """"""

        kernel_values, gk = self.kernel.compute(xk, self.mu)
        ek = dk - gk
        self.kernel.update(kernel_values, xk, ek, self.mu)

        return gk, ek
    
    def run_batch(self, x, d):
        return super().run_batch(x, d)


class KAP(KernelAdaptiveFiltersBase):
    """"""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Set of key word arguments that defines the adaptive filter.
            
            Valid keyword arguments are:

            kernel_type : str
                Defines the kernel function used (see code for more
                information on possible kernel functions).
            order : int
                Defines the number of past samples used in each
                iteration.
            step_factor : float
                Defines the step size factor used in to update the
                filter.
            Imax : int
                Maximum number of elements in the kernel function.
            args : tuple
                Arguments for the kernel function.
        """
        
        super().__init__(**kwargs)
        self.mu = kwargs['step_factor']
        self.L = kwargs['L']
        self.gamma = kwargs['gamma']
        self.kfun_kwargs['order'] = self.order
        self.kfun_kwargs['L'] = self.L
        self.kfun_kwargs['gamma'] = self.gamma
        self.kernel = KernelHandlerAP(*self.kfun_args, **self.kfun_kwargs)
        self.d_AP = np.zeros((self.L+1,))
    
    def evaluate(self, xk: np.ndarray, dk: np.ndarray):
        """"""
        self.d_AP = np.hstack((dk, self.d_AP[:-1]))
        y_AP, e_AP = self.kernel.compute(xk, self.d_AP, self.mu)

        return y_AP[0], e_AP[0]
    
    def run_batch(self, x: np.ndarray, d: np.ndarray):
        return super().run_batch(x, d)
    

class SMKAP(KernelAdaptiveFiltersBase):
    """"""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Set of key word arguments that defines the adaptive filter.
            
            Valid keyword arguments are:

            kernel_type : str
                Defines the kernel function used (see code for more
                information on possible kernel functions).
            order : int
                Defines the number of past samples used in each
                iteration.
            step_factor : float
                Defines the step size factor used in to update the
                filter.
            Imax : int
                Maximum number of elements in the kernel function.
            args : tuple
                Arguments for the kernel function.
        """
        
        super().__init__(**kwargs)
        self.mu = kwargs['step_factor']
        self.gamma_bar = kwargs['gamma_bar']
        self.L = kwargs['L']
        self.gamma = kwargs['gamma']
        self.kfun_kwargs['order'] = self.order
        self.kfun_kwargs['L'] = self.L
        self.kfun_kwargs['gamma'] = self.gamma
        self.kernel = KernelHandlerSMAP(*self.kfun_args, **self.kfun_kwargs)
        self.d_AP = np.zeros((self.L+1,))
    
    def evaluate(self, xk: np.ndarray, dk: np.ndarray):
        """"""

        self.d_AP = np.hstack((dk, self.d_AP[:-1]))
        y_AP, e_AP = self.kernel.compute(xk, self.d_AP, self.mu, self.gamma_bar)

        # y_AP, kernel_dict = self.kernel.compute(xk)
        # self.d_AP = np.hstack((dk, self.d_AP[:-1]))
        # e_AP = self.d_AP - y_AP
        # self.kernel.update(e_AP, kernel_dict, self.gamma_bar, self.mu)

        return y_AP[0], e_AP[0]
    
    def run_batch(self, x: np.ndarray, d: np.ndarray):
        return super().run_batch(x, d)


class KRS(KernelAdaptiveFiltersBase):
    """"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self):
        return super().evaluate()
    
    def run_batch(self, x: np.ndarray, d: np.ndarray):
        return super().run_batch(x, d)
