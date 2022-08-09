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
            memory_bound=None,
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
            memory_bound=None,
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

        wanted_result_exp = np.array([
            [1.00000000e+00, 1.48413159e+02],
            [1.10231764e+01, 1.98789151e+05]
        ])

        wanted_result_lin = kernel

        self.assertIsNone(np.testing.assert_allclose(self.qk_exp._postprocess_kernel(kernel), wanted_result_exp, rtol=1e-5)) # it returns None if the arrays are equal.
        self.assertIsNone(np.testing.assert_allclose(self.qk_lin._postprocess_kernel(kernel), wanted_result_lin))

    def test_image_to_circuit(self) -> None:

        image = np.array([
            [ 1,  1, -1, -1],
            [-1,  1,  1,  1],
            [ 1, -1, -1, -1],
            [ 1, -1, -1,  1]
        ], dtype=np.int8)

        qc = self.qk_exp.image_to_circuit(image)
        result = self.qk_exp.qi.execute(circuits=qc, had_transpiled=False)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_result = result.get_statevector().data.astype('float32')

        wanted_result = np.array([
            0.0, 0.25, 0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0,
            0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0,
            0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.0, 0.25
        ])

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

        self.assertIsNone(np.testing.assert_allclose(actual_result, wanted_result, rtol=1e-7))
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

        actual_result1 = self.qk_exp.kernel(X1, X2, from_quantumstate=False)
        self.qk_exp.memory_bound = 1
        actual_result2 = self.qk_exp.kernel(X1, X2, from_quantumstate=False)

        wanted_result = np.array([
            [0.16533622, 0.25482264],
            [0.25482264, 0.21157153]
        ])

        self.assertIsNone(np.testing.assert_allclose(actual_result1, wanted_result, rtol=1e-7))
        self.assertIsNone(np.testing.assert_allclose(actual_result2, wanted_result, rtol=1e-7))


if __name__ == '__main__':

    unittest.main()