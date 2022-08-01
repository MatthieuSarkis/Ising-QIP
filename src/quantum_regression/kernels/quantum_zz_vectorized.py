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
from re import A
import numpy as np
from typing import Optional
from scipy import linalg

PAULI_Z = np.array([[1, 0], [0, -1]])
HADAMARD = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

from src.quantum_regression.abstract_kernels.quantum_kernel import QUANTUM_KERNEL

class QUANTUM_ZZ_VECTORIZED(QUANTUM_KERNEL):

    def __init__(
        self,
        kernel_type: str = 'classical_copy',
        entanglement: str = "full", # ['linear', 'circular', 'full']
        reps: int = 1,
        *args,
        **kwargs
    ) -> None:
        r""" Initialize the Simulated Projected Kernel Ridge Regression from parent's init.
        Args:
            kernel_type (str): The type of kernel being used, available list: ['classical_copy', 'quantum_gaussian_sqrt', 'quantum_gaussian', 'pqk', 'linear'].

        Returns:
            None
        """

        super(QUANTUM_ZZ_VECTORIZED, self).__init__(*args, **kwargs)
        self.name = 'quantum_zz_vectorized_kernelType=' + kernel_type
        self.n_qubits: Optional[int] = None
        self.kernel_type = kernel_type
        self.entanglement = entanglement
        self.reps = reps

        self.H = None
        self.nqubits_Z = None

    def _set_n_qubits(
        self,
        X: np.ndarray
    ) -> None:

        self.n_qubits = X.shape[-1]

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

        U = self._get_unitary_vectorized(X)

        computational_basis_vector = np.zeros(shape=(2**self.n_qubits,))
        computational_basis_vector[0] = 1

        psi = np.einsum('abc,ac->ab', U, np.repeat(computational_basis_vector[np.newaxis,:], X.shape[0], axis=0))
        density_matrices = np.einsum('ac,ad->acd', psi, psi)

        return density_matrices

    def _get_hadamard(self) -> None:

        # Compute the n_qubits Hadamard
        self.H = HADAMARD
        for _ in range(self.n_qubits-1):
            self.H = np.kron(self.H, HADAMARD)

    def _get_nqubits_Z(self) -> None:

        # Compute the ZZ component of the unitary
        nqubits_Z = []
        to_tensorize = [np.eye(2) for _ in range(self.n_qubits-1)]
        for i in range(self.n_qubits):
            to_tensorize.insert(i, PAULI_Z)
            z = to_tensorize[0]
            for j in range(1, self.n_qubits):
                z = np.kron(z, to_tensorize[j])
            del to_tensorize[i]
            nqubits_Z.append(z)

        self.nqubits_Z = np.array(nqubits_Z)

    def _get_unitary_vectorized(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        r"""
        returns the ZZ unitary matrix implemented the embedding of datapoint x.
        The number of qubits is equal to the classical feature space dimension.
        """

        if self.H is None:
            self._get_hadamard()
        if self.nqubits_Z is None:
            self._get_nqubits_Z()

        if self.entanglement == "full":
            pairs = [(i, j) for i in range(self.n_qubits) for j in range(self.n_qubits) if j > i]
        elif self.entanglement == "linear":
            pairs = [(i, i+1) for i in range(self.n_qubits-1)]
        elif self.entanglement == "circular":
            pairs = [(i, i+1) for i in range(self.n_qubits-1)] + [(self.n_qubits-1, 0)]


        ## Vectorized version (batch should not be too large for memory issues)
        #xind = np.array(pairs)[:, 0]
        #yind = np.array(pairs)[:, 1]
#
        #Z = (X.reshape(-1, self.n_qubits, 1, 1) * self.nqubits_Z).sum(axis=1)
        #ZZ = np.einsum('abcd,afdh->abfch', (math.pi - X).reshape(-1, self.n_qubits, 1, 1) * self.nqubits_Z, (math.pi - X).reshape(-1, self.n_qubits, 1, 1) * self.nqubits_Z)[:, xind, yind, :, :].sum(axis=1)
#
        ## I couldn't find an easy way to vectorize this matrix exponentiation unfortunately.
        #U = np.zeros_like(Z, dtype = 'complex_')
        #for i in range(X.shape[0]):
        #    U[i] = linalg.expm(1j * (Z[i] + ZZ[i]))
#
        #unitary = np.repeat(np.identity(2**self.n_qubits)[np.newaxis,:,:], X.shape[0], axis=0)
        #for _ in range(self.reps):
        #    unitary = np.einsum('abc,ace,eg->abg', unitary, U, self.H)
#
        #return unitary

        # Loop version (no memory issues but slower)
        output = []
        for idx in range(X.shape[0]):

            x = X[idx]

            Z = np.array([x[i] * self.nqubits_Z[i] for i in range(self.n_qubits)]).sum(axis=0)
            ZZ = np.array([((math.pi - x[i]) * self.nqubits_Z[i]) @ ((math.pi - x[j]) * self.nqubits_Z[j]) for (i, j) in pairs]).sum(axis=0)
            U = linalg.expm(1j * (Z + ZZ))

            unitary = np.identity(2**self.n_qubits)
            for _ in range(self.reps):
                unitary = unitary @ U @ self.H

            output.append(unitary)

        output = np.array(output)

        return output