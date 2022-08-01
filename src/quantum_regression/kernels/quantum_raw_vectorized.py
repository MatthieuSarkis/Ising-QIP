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

import math
import numpy as np
from typing import Optional

from src.kernel_ridge_regression.abstract_kernels.quantum_kernel import QUANTUM_KERNEL

class QUANTUM_RAW_VECTORIZED(QUANTUM_KERNEL):

    def __init__(
        self,
        kernel_type: str = 'classical_copy',
        *args,
        **kwargs
    ) -> None:
        r""" Initialize the Simulated Projected Kernel Ridge Regression from parent's init.

        Args:
            kernel_type (str): The type of kernel being used, available list: ['classical_copy', 'quantum_gaussian_sqrt', 'quantum_gaussian', 'pqk', 'linear'].

        Returns:
            None
        """

        super(QUANTUM_RAW_VECTORIZED, self).__init__(*args, **kwargs)
        self.name = 'quantum_raw_vectorized_kernelType=' + kernel_type
        self.n_qubits: Optional[int] = None
        self.kernel_type = kernel_type

    def _set_n_qubits(
        self,
        X: np.ndarray
    ) -> None:

        self.n_qubits = math.ceil(np.log2(X.shape[-1]))

    def _postprocess_kernel(
        self,
        kernel: np.ndarray
    ) -> np.ndarray:

        if self.kernel_type == 'classical_copy':
            K = np.exp(- kernel / (2 * self.sigma**2))
        elif self.kernel_type == 'quantum_gaussian_sqrt':
            K = np.exp(- kernel / (2 * self.sigma**2))
        elif self.kernel_type == 'quantum_gaussian':
            K = np.exp(- kernel / self.sigma**4)
        elif self.kernel_type == 'pqk':
            K = np.exp(- kernel / self.sigma**4)
        elif self.kernel_type == 'linear':
            K = kernel

        return K

    def _scalar_product(
        self,
        rho1: np.ndarray,
        rho2: np.ndarray
    ) -> float:

        if self.kernel_type == 'classical_copy':
            # Tr[rho1] + Tr[rho2] - 2 * Tr[rho1.rho2]**0.5
            K = np.repeat(np.einsum('abb->a', rho1)[:, np.newaxis], rho2.shape[0], axis=1) + np.repeat(np.einsum('abb->a', rho2)[np.newaxis, :], rho1.shape[0], axis=0) - 2 * np.einsum('agc,dcg->ad', rho1, rho2) ** 0.5

        elif self.kernel_type == 'quantum_gaussian_sqrt':
            # -2 * Tr[rho1.rho2]**0.5
            K = -2 * np.sqrt(np.einsum('abc,dcb->ad', rho1, rho2))

        elif self.kernel_type == 'quantum_gaussian':
            # - Tr[rho1.rho2]
            K = - np.einsum('abc,dcb->ad', rho1, rho2)

        elif self.kernel_type == 'pqk':
            # ||rho1 - rho2||_F ** 2 = Tr[rho1**2] + Tr[rho2**2] - 2 * Tr[rho1.rho2]
            tr_rho1_squared = np.repeat(np.einsum('abc,acb->a', rho1, rho1)[:,np.newaxis], rho2.shape[0], axis=1)
            tr_rho2_squared = np.repeat(np.einsum('abc,acb->a', rho2, rho2)[np.newaxis,:], rho1.shape[0], axis=0)
            tr_rho1_rho2 = np.einsum('abc,dcb->ad', rho1, rho2)
            K = tr_rho1_squared + tr_rho2_squared - 2 * tr_rho1_rho2

        elif self.kernel_type == 'linear':
            # Tr[rho1.rho2] followed by no postprocessing
            K = np.einsum('abc,dcb->ad', rho1, rho2)

        return K

    def _get_density_matrices(
        self,
        X: np.ndarray,
    ) -> None:

        X_padded = np.pad(array=X, pad_width=((0, 0), (0, 2**self.n_qubits - X.shape[1])), mode='constant', constant_values=((0., 0.), (0., 0.)))
        X_padded = X_padded / np.linalg.norm(X_padded, axis=1).reshape((-1, 1))
        density_matrices = np.einsum('ac,ad->acd', X_padded, X_padded)
        return density_matrices