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

r"""
Implementation of the Quantum Kernel Ridge Regression.
"""

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.result.counts import Counts
from qiskit.utils.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit.result import Result
from math import ceil, log2

from src.kernel_ridge_regression.abstract_kernels.qiskit_kernel import QiskitKernel
from src.kernel_ridge_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression


class Quantum_Kernel(KernelRidgeRegression, QiskitKernel):
    r"""Class implementing the quantum kernel. The kernel can either be linear or exponential
    depending on whether a value of gamma has been provided to the constructor or not.
    """

    def __init__(
        self,
        memory_bound: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        r"""Constructor for the quantum kernel class."""

        super(Quantum_Kernel, self).__init__(*args, **kwargs)
        self.name = 'quantum_kernel'
        self.num_qubits: Optional[int] = None
        self.memory_bound = memory_bound


    def kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        from_quantumstate: bool = False
    ) -> np.ndarray:
        r""" Compute the Gaussian Kernel.
        Args:
            X1 (np.ndarray): First batch of 2d images.
            X2 (np.ndarray): Second batch of 2d images.
            from_quantumstate (bool): Whether of not to use preprocessed image data
            which has already been mapped to a quantum state.
        Returns:
            (np.ndarray): Quantum Gram matrix associated to the two batches of data X1 and X2.
        """

        N1 = X1.shape[0]
        N2 = X2.shape[0]

        # In case the batches of data are not too large (i.e. do not exceed the self.memory_bound attribute)
        # one computes the Gram matrix entirely.
        if (self.memory_bound is None) or (N1 <= self.memory_bound and N2 <= self.memory_bound):

            if from_quantumstate:
                gram = np.einsum('ab,cb->ac', X1, X2)

            else:
                num_qubits = 2 * ceil(log2(X1.shape[-1])) + 1

                bounds1 = []
                for i in range(X1.shape[0]):
                    q = self.image_to_circuit(image=X1[i])
                    bounds1.append(q)

                bounds2 = []
                for i in range(X2.shape[0]):
                    q = self.image_to_circuit(image=X2[i])
                    bounds2.append(q)

                circuits = []
                for c1 in bounds1:
                    for c2 in bounds2:
                        c = c2.compose(c1.inverse())
                        if self._backend_type == 'IBMQ':
                            c = c.measure_all(inplace=False)
                        circuits.append(c)

                #qc = transpile(circuits, self._backend)

                if self._backend_type == 'IBMQ':

                    if self.mitigate:
                        qr = QuantumRegister(self.n_qubits)
                        meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
                        #cal_qc = transpile(meas_calibs, self.qi.backend)
                        #qc = cal_qc + qc
                        qc = meas_calibs + qc

                        results = self.qi.execute(circuits=qc, had_transpiled=False)

                        # Split the results into to Result objects for calibration and kernel computation
                        cal_res = Result(backend_name= results.backend_name,
                            backend_version= results.backend_version,
                            qobj_id= results.qobj_id,
                            job_id= results.job_id,
                            success= results.success,
                            #results = results.results[:len(cal_qc)]
                            results = results.results[:len(meas_calibs)]
                        )

                        data_res = Result(backend_name= results.backend_name,
                            backend_version= results.backend_version,
                            qobj_id= results.qobj_id,
                            job_id= results.job_id,
                            success= results.success,
                            #results = results.results[len(cal_qc):]
                            results = results.results[len(meas_calibs):]
                        )

                        # Apply measurement calibration and computer the calibration filter
                        meas_filter = CompleteMeasFitter(cal_res, state_labels, circlabel='mcal').filter

                        # Apply the calibration filter to the results and get the counts
                        mitigated_results = meas_filter.apply(data_res)
                        self.counts = mitigated_results.get_counts()

                    else:
                        #result = self.qi.execute(circuits=qc, had_transpiled=True)
                        result = self.qi.execute(circuits=circuits, had_transpiled=False)
                        counts = result.get_counts()

                    # Handle the case of a batch containing a single image
                    if type(counts) == Counts:
                        counts = [counts]

                    gram = [count[num_qubits*'0'] / self.shots for count in counts]
                    gram = np.array(gram)

                else:
                    computational_basis_vector = np.zeros(shape=(2**num_qubits,))
                    computational_basis_vector[0] = 1
                    #result = self.qi.execute(circuits=qc, had_transpiled=True)
                    result = self.qi.execute(circuits=circuits, had_transpiled=False)
                    statevector = [np.real(result.get_statevector(i)) for i in range(len(circuits))]
                    statevector = np.stack(statevector)
                    gram = np.einsum('a,ba->b', computational_basis_vector, statevector)**2 # One has to square the amplitudes to get the probabilities

                gram = gram.reshape((X1.shape[0], X2.shape[0]))

            gram = np.absolute(gram)**2 # modulus square |<psi1|psi2>|^2
            gram = self._postprocess_kernel(gram)
            return gram

        # In case the batches of data exceed the bound allowed by the RAM (specified by self.memory_bound),
        # split the batches into smaller batches.
        else:

            gram = np.zeros(shape=(N1, N2))

            for i in range(0, N1, self.memory_bound):
                for j in range(0, N2, self.memory_bound):

                    X1_temp = X1[i:i+self.memory_bound]
                    X2_temp = X2[j:j+self.memory_bound]
                    n1 = X1_temp.shape[0]
                    n2 = X2_temp.shape[0]
                    gram[i:i+n1, j:j+n2] = self.kernel(X1_temp, X2_temp)

            return gram


    def image_to_circuit(
        self,
        image: np.ndarray
    ) -> QuantumCircuit:
        r"""Associates a quantum circuit to an image. The output of the circuit is the encoded image.
        Args:
            image (np.mdarray): Single image whose quantum circuit representation one is computing.
        Returns:
            (QuantumCircuit): Quatum circuit representation of image.
        """

        image_size = image.shape[0]
        side_qubits = ceil(log2(image_size))
        image_qubits = 2 * side_qubits
        self.num_qubits = image_qubits + 1

        image_register = QuantumRegister(image_qubits, 'position')
        spin_register = QuantumRegister(1, 'spin')

        qc = QuantumCircuit(spin_register, image_register)

        qc.i(0)
        for i in range(1, image_register.size + 1):
            qc.h(i)

        qc.barrier()

        for i in range(image_size):
            for j in range(image_size):

                if image[i, j] == 1:

                    binarized_i = self.binary_formatting(digit=i, n_bits=side_qubits, reverse=False)
                    binarized_j = self.binary_formatting(digit=j, n_bits=side_qubits, reverse=False)

                    flip_idx = []

                    for ii in range(side_qubits):
                        if binarized_i[ii] == '1':
                            flip_idx.append(ii+1)
                            qc.x(ii+1)

                    for jj in range(side_qubits):
                        if binarized_j[jj] == '1':
                            flip_idx.append(jj+side_qubits+1)
                            qc.x(jj+side_qubits+1)

                    qc.mcx(list(range(1, image_qubits+1)), 0, mode='noancilla')

                    for q in flip_idx:
                        qc.x(q)

                    qc.barrier()

        return qc

    @staticmethod
    def binary_formatting(
        digit: int,
        n_bits: int,
        reverse: bool = False,
    ) -> str:
        r"""
        Args:
            digit (int): number whose binary representation we are computing
            n_bits (int): number of bits used in the binary representation (to handle left trailing zeros)
            reverse (bool): optionally return the binary representation of digit in reverse order (for qiskit convention)
        Returns:
            (str): binary representation of digit
        """

        binary = format(digit, '0{}b'.format(n_bits))

        if reverse:
            binary = binary[::-1]

        return binary

    def _postprocess_kernel(
        self,
        kernel: np.ndarray
    ) -> np.ndarray:


        if self.gamma is not None:
            kernel = np.exp(-self.gamma * kernel)

        return kernel