# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import string
from typing import Union, Optional
import itertools
from scipy import linalg
from typing import Dict

from src.quantum_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression

class QUANTUM_KERNEL(KernelRidgeRegression):

    def __init__(
        self,
        n_non_traced_out_qubits: Union[int, str] = 'all',
        *args,
        **kwargs,
    ) ->None:
        """ Initialize the Simulated Projected Kernel Ridge Regression from parent's init. """

        super(QUANTUM_KERNEL, self).__init__(*args, **kwargs)
        self.name = 'quantum_kernel'
        self.n_qubits: Optional[int] = None
        self.n_non_traced_out_qubits = n_non_traced_out_qubits
        self.hash_density_matrices: Dict[int, np.ndarray] = {}
        self.hash_partial_traces: Dict[int, np.ndarray] = {}

    def kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the Quantum Kernel between two datasets X1 and X2.
        Args:
            X1: A dataset.
            X2: A dataset.
        Returns:
            The Quantum Kernel matrix.
        """

        self._set_n_qubits(X1)

        if self.n_qubits < 2:
            raise ValueError("The system should be composed of at least 2 qubits.")

        key1 = hash(X1.tobytes())
        if not key1 in self.hash_density_matrices:
            self.hash_density_matrices[key1] = self._get_density_matrices(X1)
        rho1 = self.hash_density_matrices[key1]

        key2 = hash(X2.tobytes())
        if not key2 in self.hash_density_matrices:
            self.hash_density_matrices[key2] = self._get_density_matrices(X2)
        rho2 = self.hash_density_matrices[key2]

        if self.n_non_traced_out_qubits == 'all':
            kernel = self._scalar_product(rho1, rho2)

        else:
            kernel = np.zeros(shape=(X1.shape[0], X2.shape[0]))

            for to_keep in itertools.combinations(range(self.n_qubits), self.n_non_traced_out_qubits):

                key1 = hash((to_keep, rho1.tobytes()))
                if not key1 in self.hash_partial_traces:
                    self.hash_partial_traces[key1] = self._partial_traces_vectorized(rho1, to_keep)
                partial_rho1 = self.hash_partial_traces[key1]

                key2 = hash((to_keep, rho2.tobytes()))
                if not key2 in self.hash_partial_traces:
                    self.hash_partial_traces[key2] = self._partial_traces_vectorized(rho2, to_keep)
                partial_rho2 = self.hash_partial_traces[key2]

                kernel = kernel + self._scalar_product(partial_rho1, partial_rho2)

        kernel = self._postprocess_kernel(kernel)

        return kernel

    def _set_n_qubits(
        self,
        X: np.ndarray
    ) -> None:

        raise NotImplementedError

    def _postprocess_kernel(
        self,
        kernel: np.ndarray
    ) -> np.ndarray:

        raise NotImplementedError

    @staticmethod
    def _scalar_product(
        rho1: np.ndarray,
        rho2: np.ndarray
    ) -> float:
        r'''
        Inputs:
            rho1 (np.ndarray): 1st dimension: batch, (2nd, 3rd) dimensions: density matrix
            rho2 (np.ndarray): 1st dimension: batch, (2nd, 3rd) dimensions: density matrix
        '''

        raise NotImplementedError

    def _get_density_matrices(
        self,
        X: np.ndarray,
    ) -> None:

        raise NotImplementedError

    def _partial_traces_vectorized(
        self,
        density_matrices,
        qubits_to_keep
    ) -> np.ndarray:
        r''' Efficient method to compute the partial traces for a batch of density matrices.
        Inputs:
            density_matrices (np.ndarray): 1st dim: batch dimension, (2nd, 3rd) dim: density matrix
            qubits_to_keep (list): list of qubit indices not to trace over
        Returns:
            (np.ndarray): 1st dim: batch dimension, (2nd, 3rd) dim: partial trace of density matrix
        '''

        if len(qubits_to_keep) == self.n_qubits:
            return density_matrices

        to_trace = [i for i in range(self.n_qubits) if i not in qubits_to_keep]

        source_idx = list(string.ascii_lowercase[1: 2 * self.n_qubits + 1])
        target_idx = list(string.ascii_lowercase[1: 2 * self.n_qubits + 1])

        for i in to_trace:
            source_idx[self.n_qubits + i] = source_idx[i]
        for j in sorted(to_trace + [self.n_qubits + i for i in to_trace], reverse=True):
            del target_idx[j]

        source_idx = ['a'] + source_idx
        target_idx = ['a'] + target_idx
        einsum_rule = ''.join(source_idx + ['-', '>'] + target_idx)

        partial_traces = np.einsum(einsum_rule, density_matrices.reshape(tuple([density_matrices.shape[0]] + [2 for _ in range(2*self.n_qubits)])))
        partial_traces = partial_traces.reshape((partial_traces.shape[0], 2**len(qubits_to_keep), 2**len(qubits_to_keep)))

        return partial_traces

    @staticmethod
    def von_neumann_entropy(
        rho: np.ndarray
    ) -> np.ndarray:
        r""" Computes the von neumann entropy of a batch of density matrices
        Args:
            rho (np.ndarray): batch of density matrices.
        Returns:
            (np.ndarray): batch of the corresponding von neumann entropies.
        """

        N = rho.shape[0]
        entropy = np.zeros(shape=(N,))

        for i in range(N):
            entropy[i] = -np.trace(rho[i] @ linalg.logm(rho[i]))

        return entropy