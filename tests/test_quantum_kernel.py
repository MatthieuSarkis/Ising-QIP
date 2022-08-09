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

from collections import OrderedDict
import numpy as np
import unittest

from src.kernel_ridge_regression.kernels.quantum_kernel import Quantum_Kernel


class Test_Quantum_Kernel(unittest.TestCase):

    def setUp(self) -> None:

        self.qk_lin = Quantum_Kernel(
            gamma=None,
            ridge_parameter=1.0,
            backend_type='simulator',
            backend_name='statevector_simulator',
            mitigate=False,
            seed= 42,
            shots= None,
            hub='ibm-q',
            group='open',
            project='main',
            job_name='ising_qip',
        )

        self.qk_exp = Quantum_Kernel(
            gamma=1.0,
            ridge_parameter=1.0,
            backend_type='simulator',
            backend_name='statevector_simulator',
            mitigate=False,
            seed= 42,
            shots= None,
            hub='ibm-q',
            group='open',
            project='main',
            job_name='ising_qip',
        )

    def test_binary_formatting(self) -> None:

        result1 = Quantum_Kernel.binary_formatting(0, 8, False)
        result2 = Quantum_Kernel.binary_formatting(3, 5, False)
        result3 = Quantum_Kernel.binary_formatting(25, 11, True)

        self.assertEqual(result1, '00000000')
        self.assertEqual(result2, '00011')
        self.assertEqual(result3, '10011000000')

    def test_postprocess_kernel(self) -> None:

        kernel = np.array([
            [1.0, 3.5],
            [2.2, 7.1]
        ])
        result_exp = np.array([
            [0.36787944, 0.03019738],
            [0.11080316, 0.0008251 ]
        ])
        result_lin = kernel

        self.assertIsNone(np.testing.assert_allclose(self.qk_exp._postprocess_kernel(kernel), result_exp, rtol=1e-5)) # it returns None if the arrays are equal.
        self.assertIsNone(np.testing.assert_allclose(self.qk_lin._postprocess_kernel(kernel), result_lin)) # it returns None if the arrays are equal.

    def test_image_to_circuit(self) -> None:

        image = np.array([
            [ 1,  1, -1, -1],
            [-1,  1,  1,  1],
            [ 1, -1, -1, -1],
            [ 1, -1, -1,  1]
        ], dtype=np.int8)

        qc = self.qk_exp.image_to_circuit(image)
        result = self.qk_exp.qi.execute(circuits=qc, had_transpiled=False)

        actual_result = result.get_statevector().data.astype('float32')
        wanted_result = np.array([0.0, 0.25, 0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0,0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0,0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.0, 0.25])

        self.assertIsNone(np.testing.assert_allclose(actual_result, wanted_result, rtol=1e-7))

        wanted_result2 = OrderedDict([
            ('h', 32),
            ('u3', 30),
            ('cu1', 16),
            ('barrier', 9),
            ('rcccx', 8),
            ('rcccx_dg', 8),
            ('c3sx', 8),
            ('u2', 4),
            ('id', 1)
        ])

        self.assertDictEqual(qc.decompose().count_ops(), wanted_result2)

    def test_kernel(self) -> None:

        X1 = np.array([
            [[ 1, -1, -1, -1],
             [-1, -1,  1, -1],
             [ 1,  1,  1,  1],
             [ 1,  1,  1,  1]],

            [[ 1,  1,  1,  1],
             [-1, -1,  1,  1],
             [ 1, -1,  1,  1],
             [ 1,  1,  1,  1]]], dtype=np.int8)

        X2 = np.array([
            [[ 1,  1,  1,  1],
             [-1, -1, -1, -1],
             [-1, -1, -1,  1],
             [ 1,  1,  1,  1]],

            [[ 1, -1, -1, -1],
             [-1, -1,  1,  1],
             [ 1, -1, -1, -1],
             [ 1,  1,  1,  1]]], dtype=np.int8)

        actual_result = self.qk_exp.kernel(X1, X2)
        wanted_result = np.array([
            [0.90473525, 0.72876333],
            [0.72876333, 0.79979172]
        ])

        self.assertIsNone(np.testing.assert_allclose(actual_result, wanted_result, rtol=1e-7))


if __name__ == '__main__':

    unittest.main()