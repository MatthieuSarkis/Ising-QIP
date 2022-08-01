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

from abc import abstractmethod
import itertools
import numpy as np
from typing import Dict, Optional, Union, Tuple

from src.quantum_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression

class ProjectedQuantumKernel(KernelRidgeRegression):
    r"""
    Parent class for all projected quantum kernels
    Children classes have to implement methods related to:
    1) How to compute the density matrix corresponding to a data point
    2) How to combine partial traces of the dencity matrices to form the kernel
    3) Is the model linear, exponential, etc... (how to 'postprocess' the kernel)
    """

    def __init__(
        self,
        n_non_traced_out_qubits: Union[int, str] = 1,
        precompute: bool = False,
        *args,
        **kwargs,
    ) ->None:
        """ Initialize the Projected Kernel Ridge Regression from parent's init.

        Args:
            n_non_traced_out_qubits: size of the system after the partial traces.
            Can take the string value 'all' to keep all qubits.
        """

        super(ProjectedQuantumKernel, self).__init__(*args, **kwargs)
        self.hash_density_matrix: Dict[int, np.ndarray] = {}
        self.hash_partial_traces: Dict[int, np.ndarray] = {}
        self.n_qubits: Optional[int] = None
        self.n_non_traced_out_qubits = n_non_traced_out_qubits
        self.precompute = precompute

    def kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the Projected Quantum Kernel between two datasets X1 and X2.
        Args:
            X1: A dataset.
            X2: A dataset.
        Returns:
            The Projected Quantum Kernel matrix.
        """

        M = X1.shape[0]
        N = X2.shape[0]

        self._set_n_qubits(X1)

        if self.n_non_traced_out_qubits == 'all':
            self.n_non_traced_out_qubits = self.n_qubits

        if self.n_qubits < 2:
            raise ValueError("The system should be composed of at least 2 qubits.")

        if self.precompute:
            key1 = [hash((i, X1.tobytes())) for i in range(M)]
            if not all(k in self.hash_density_matrix for k in key1):
                RHO1 = self._get_density_matrices(X1)
                for ind,key in enumerate(key1):
                    if not key in self.hash_density_matrix:
                        self.hash_density_matrix[key] = RHO1[ind]

            key2 = [hash((i, X2.tobytes())) for i in range(N)]
            if not all(k in self.hash_density_matrix for k in key2):
                RHO2 = self._get_density_matrices(X2)
                for ind,key in enumerate(key2):
                    if not key in self.hash_density_matrix:
                        self.hash_density_matrix[key] = RHO2[ind]

        kernel = np.zeros(shape=(M, N))

        for i in range(M):

            key1 = hash((i, X1.tobytes()))
            if not key1 in self.hash_density_matrix:
                self.hash_density_matrix[key1] = self._get_density_matrix(X1[i,:])
            rho_x1 = self.hash_density_matrix[key1]

            for j in range(N):

                key2 = hash((j, X2.tobytes()))
                if not key2 in self.hash_density_matrix:
                    self.hash_density_matrix[key2] = self._get_density_matrix(X2[j,:])
                rho_x2 = self.hash_density_matrix[key2]

                # Compute the partial trace over all qubits and add the result to the kernel matrix
                for t in itertools.combinations(range(self.n_qubits), self.n_non_traced_out_qubits):

                    key1 = hash((t, rho_x1.tobytes()))
                    if not key1 in self.hash_partial_traces:
                        self.hash_partial_traces[key1] = self.partial_trace(rho_x1, t)

                    key2 = hash((t, rho_x2.tobytes()))
                    if not key2 in self.hash_partial_traces:
                        self.hash_partial_traces[key2] = self.partial_trace(rho_x2, t)

                    kernel[i, j] += self._scalar_product(self.hash_partial_traces[key1], self.hash_partial_traces[key2])

        return self._postprocess_kernel(kernel)

    @abstractmethod
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

        raise NotImplementedError

    def _get_density_matrix(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        raise NotImplementedError

    def _get_density_matrices(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        raise NotImplementedError

    def partial_trace(
        self,
        rho: np.ndarray,
        qubits_to_keep: Tuple[int],
    ) -> np.ndarray:
        """ Calculate the partial trace for qubit system. Code made by Huo Chen neversakura and accessible at: https://gist.github.com/neversakura/d6a60b4bb2990d252e9e89e5629d5553.
        Args:
            rho: Density matrix
            qubit_to_keep: The list of index of qubit to be kept after taking the trace
            Density matrix after taking partial trace.
        """

        if len(qubits_to_keep) == self.n_qubits:
            return rho

        num_qubit = int(np.log2(rho.shape[0]))
        qubit_axis = [(i, num_qubit + i) for i in range(num_qubit) if i not in qubits_to_keep]
        minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
        minus_qubit_axis = [(q[0] - m[0], q[1] - m[1]) for q, m in zip(qubit_axis, minus_factor)]

        rho_res = np.reshape(rho, [2, 2] * num_qubit)
        qubit_left = num_qubit - len(qubit_axis)

        for i, j in minus_qubit_axis:
            rho_res = np.trace(rho_res, axis1=i, axis2=j)

        if qubit_left > 1:
            rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)

        return rho_res