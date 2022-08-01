# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi (https://github.com/adelshb).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Qiskit implementation of the Projected Quantum Kernel (ref: arXiv:2011.01938). """

import math
import numpy as np
from numpy import ndarray
from typing import List, Optional

from qiskit import transpile, QuantumRegister
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.result import Result

from src.quantum_regression.abstract_kernels.qiskit_kernel import QiskitKernel
from src.quantum_regression.abstract_kernels.projected_quantum_kernel import ProjectedQuantumKernel

class QPQK_ZZ(ProjectedQuantumKernel, QiskitKernel):
    """ Qiskit implementation of the Projected Quantum Kernekl for Kernel Ridge Regression with the ZZ feature map."""

    def __init__(
        self,
        reps: Optional[int] = 1,
        entanglement: Optional[str] = 'full',
        *args,
        **kwargs,
    ) ->None:
        """ Initialize the Projected Kernel Ridge Regression from parent's init. """

        super(QPQK_ZZ, self).__init__(*args, **kwargs)
        self.name = 'qpqk_zz'
        self.reps = reps
        self.entanglement = entanglement

    def _get_density_matrix(
        self,
        x: ndarray,
    ) -> List[ndarray]:
        """ Get the density matrices of the dataset X.
        Args:
            X: A dataset.
        Returns:
            A List of density matrices.
        """

        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.reps,
            entanglement=self.entanglement
            ).decompose()

        qc = state_tomography_circuits(feature_map.assign_parameters(x),[i for i in range(self.n_qubits)])

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
            job = meas_filter.apply(data_res)
            tomo_fitter = StateTomographyFitter(job, qc[len(cal_qc):])
        else:
            job = self.qi.execute(circuits=qc, had_transpiled=True)
            tomo_fitter = StateTomographyFitter(job, qc)

        rho = tomo_fitter.fit(method='lstsq')
        return rho

    def _get_density_matrices(
        self,
        X: np.ndarray,
    ) -> List[ndarray]:

        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.reps,
            entanglement=self.entanglement
            ).decompose()

        num_meas = len(state_tomography_circuits(feature_map.assign_parameters(X[0,:]),[i for i in range(self.n_qubits)]))

        qc = sum([state_tomography_circuits(feature_map.assign_parameters(x),[i for i in range(self.n_qubits)]) for x in X],[])

        if self.mitigate:
            qr = QuantumRegister(self.n_qubits)
            meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
            cal_qc = transpile(meas_calibs, self.qi.backend)
            # qc = cal_qc + qc
            qc = meas_calibs + qc

            results = self.qi.execute(circuits=qc, had_transpiled=False)

            # Split the results into to Result objects for calibration and kernel computation
            cal_res = Result(backend_name= results.backend_name,
                backend_version= results.backend_version,
                qobj_id= results.qobj_id,
                job_id= results.job_id,
                success= results.success,
                results = results.results[:len(cal_qc)]
            )

            # Apply measurement calibration and computer the calibration filter
            meas_filter = CompleteMeasFitter(cal_res, state_labels, circlabel='mcal').filter

            # Apply the calibration filter to the results and get the counts
            tomo_fitters = []
            for i in range(X.shape[0]):
                data_res = Result(backend_name= results.backend_name,
                    backend_version= results.backend_version,
                    qobj_id= results.qobj_id,
                    job_id= results.job_id,
                    success= results.success,
                    results = results.results[len(cal_qc) + i * num_meas:len(cal_qc) + (i+1) * num_meas]
                )
                job = meas_filter.apply(data_res)
                tomo_fitters.append(StateTomographyFitter(job, qc[len(cal_qc) + i * num_meas:len(cal_qc) + (i+1) * num_meas]))
        else:
            results = self.qi.execute(circuits=qc, had_transpiled=True)
            tomo_fitters = []
            for i in range(X.shape[0]):
                job = Result(backend_name= results.backend_name,
                    backend_version= results.backend_version,
                    qobj_id= results.qobj_id,
                    job_id= results.job_id,
                    success= results.success,
                    results = results.results[len(cal_qc) + i * num_meas:len(cal_qc) + (i+1) * num_meas]
                )
                tomo_fitters.append(StateTomographyFitter(job, qc[len(cal_qc) + i * num_meas:len(cal_qc) + (i+1) * num_meas]))

        RHO = [tomo_fitter.fit(method='lstsq') for tomo_fitter in tomo_fitters]
        return RHO

    def _set_n_qubits(
        self,
        X: np.ndarray
    ) -> None:

        self.n_qubits = X.shape[1]

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

        return np.exp(- kernel / self.sigma**4)