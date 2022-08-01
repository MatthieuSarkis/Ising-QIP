# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi (https://github.com/adelshb) and Matthieu Sarkis (https://github.com/MatthieuSarkis).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Qiskit implementation of the Projected Quantum Kernel (ref: arXiv:2011.01938). """

from typing import Dict, List

import math
import numpy as np
from numpy import linalg as LA
from numpy import ndarray

from qiskit import transpile, QuantumRegister, QuantumCircuit
from qiskit.result.counts import Counts
from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.result import Result

from src.quantum_regression.abstract_kernels.qiskit_kernel import QiskitKernel
from src.quantum_regression.abstract_kernels.projected_quantum_kernel import ProjectedQuantumKernel

class QPQK_RAW(ProjectedQuantumKernel, QiskitKernel):
    """ Qiskit implementation of the Projected Quantum Kernekl for Kernel Ridge Regression with raw data encoded in quantum states."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) ->None:
        """ Initialize the Projected Kernel Ridge Regression from parent's init. """

        super().__init__(*args, **kwargs)
        self.name = 'qpqk_raw'

    def _get_density_matrix(
        self,
        x: ndarray,
    ) -> List[ndarray]:
        """ Get the density matrix of the data point x.
        Args:
            X: A data point.
        Returns:
            The density matrix.
        """

        feature_map = RawFeatureVector(2**self.n_qubits)

        # Apply the data value as amplitude of the quantum state of the right size
        q = QuantumCircuit(self.n_qubits, self.n_qubits)

        state = np.zeros(shape=(2**self.n_qubits,))
        state[:x.shape[0]] = x
        state = state / LA.norm(state)
        q.append(feature_map.assign_parameters(state), qargs=[i for i in range(self.n_qubits)])

        q.measure([i for i in range(self.n_qubits)], [i for i in range(self.n_qubits)])

        qc = transpile(q, self.qi.backend)

        if self.mitigate:
            # Add measurement calibration circuits in front
            qr = QuantumRegister(self.n_qubits)
            meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
            cal_qc = transpile(meas_calibs, self.qi.backend)
            qc = cal_qc + qc

            # Run the experiment with specified backend
            results = self.qi.execute(circuits=qc, had_transpiled=True)

            # Split the results into to Result objects for calibration and kernel computation
            cal_res = Result(backend_name= results.backend_name,
                backend_version= results.backend_version,
                qobj_id= results.qobj_id,
                job_id= results.job_id,
                success= results.success,
                results = results.results[:len(cal_qc)]
            )

            data_res = Result(backend_name= results.backend_name,
                backend_version= results.backend_version,
                qobj_id= results.qobj_id,
                job_id= results.job_id,
                success= results.success,
                results = results.results[len(cal_qc):]
            )

            # Apply measurement calibration and computer the calibration filter
            meas_filter = CompleteMeasFitter(cal_res, state_labels, circlabel='mcal').filter

            # Apply the calibration filter to the results and get the counts
            mitigated_results = meas_filter.apply(data_res)
            self.counts = mitigated_results.get_counts()
        else:
            self.counts = self.qi.execute(circuits=qc, had_transpiled=True).get_counts()

        if type(self.counts) == Counts:
            self.counts = [self.counts]

        # Get the density matrix from the counts
        RHO = [self._counts_to_density_matrix(self.counts[ind], 2**self.n_qubits) for ind in range(len(self.counts))]
        return RHO[0]

    def _get_density_matrices(
        self,
        X: ndarray,
    ) -> List[ndarray]:
        """ Get the density matrices of the dataset X.
        Args:
            X: A dataset.
        Returns:
            A List of density matrices.
        """

        bounds = []
        feature_map = RawFeatureVector(2**self.n_qubits)

        # Arrange a list of cricuits to be executed and where the normalized data point is the amplitude of the quantum state
        for i in range(X.shape[0]):
            q = QuantumCircuit(self.n_qubits, self.n_qubits)

            state = np.zeros(shape=(2**self.n_qubits,))
            state[:X[i, :].shape[0]] = X[i, :]
            state = state / LA.norm(state)
            q.append(feature_map.assign_parameters(state), qargs=[i for i in range(self.n_qubits)])

            q.measure([i for i in range(self.n_qubits)], [i for i in range(self.n_qubits)])
            bounds.append(q)

        qc = transpile(bounds, self.qi.backend)

        if self.mitigate:
            qr = QuantumRegister(self.n_qubits)
            meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
            cal_qc = transpile(meas_calibs, self.qi.backend)
            qc = cal_qc + qc

            results = self.qi.execute(circuits=qc, had_transpiled=True)

            # Split the results into to Result objects for calibration and kernel computation
            cal_res = Result(backend_name= results.backend_name,
                backend_version= results.backend_version,
                qobj_id= results.qobj_id,
                job_id= results.job_id,
                success= results.success,
                results = results.results[:len(cal_qc)]
            )

            data_res = Result(backend_name= results.backend_name,
                backend_version= results.backend_version,
                qobj_id= results.qobj_id,
                job_id= results.job_id,
                success= results.success,
                results = results.results[len(cal_qc):]
            )

            # Apply measurement calibration and computer the calibration filter
            meas_filter = CompleteMeasFitter(cal_res, state_labels, circlabel='mcal').filter

            # Apply the calibration filter to the results and get the counts
            mitigated_results = meas_filter.apply(data_res)
            self.counts = mitigated_results.get_counts()
        else:
            self.counts = self.qi.execute(circuits=qc, had_transpiled=True).get_counts()

        if type(self.counts) == Counts:
            self.counts = [self.counts]

        RHO = [self._counts_to_density_matrix(self.counts[ind], 2**self.n_qubits) for ind in range(len(self.counts))]
        return RHO

    @staticmethod
    def _counts_to_density_matrix(
        counts: Dict[str,int],
        num_features: int,
    ) -> ndarray:
        """ Get the density matrix from data counts.
        Args:
            counts: A dictionary of counts.
            num_features: The number of features in the dataset.
        Returns:
            A density matrix.
        """

        statevector = np.zeros(shape=(num_features,))
        for key, value in counts.items():
            statevector[int(key, 2)] = value
        statevector = statevector / np.linalg.norm(statevector)
        return np.outer(statevector, statevector)

    def _set_n_qubits(
        self,
        X: np.ndarray
    ) -> None:

        self.n_qubits = math.ceil(np.log2(X.shape[1]))

    @staticmethod
    def _scalar_product(
        rho1: np.ndarray,
        rho2: np.ndarray
    ) -> float:

        # quantum quartic
        return - np.einsum('abc,dcb->ad', rho1, rho2)
        # return np.linalg.norm(rho1 - rho2, ord='fro') ** 2

    def _postprocess_kernel(
        self,
        kernel: np.ndarray
    ) -> np.ndarray:

        return np.exp(- kernel / self._sigma**4)
