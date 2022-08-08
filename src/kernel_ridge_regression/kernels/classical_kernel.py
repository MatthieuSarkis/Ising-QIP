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
Implementation of the Gaussian Kernel Ridge Regression.
"""

import numpy as np
from scipy.spatial.distance import cdist

from src.kernel_ridge_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression

class Gaussian(KernelRidgeRegression):
    """ Gaussian Kernel for Kernel Ridge Regression. """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        r""" Initialize the Gaussian Kernel Ridge Regression from parent's init. """

        super(Gaussian, self).__init__(*args, **kwargs)
        self.name = 'gaussian'

    def kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the Gaussian Kernel.
        Args:
            X1 (np.ndarray): First batch of 1-D array data vector.
            X2 (np.ndarray): Second batch of 1-D array data vector.
        Returns:
            (np.ndarray): Gram matrix associated to the two batches of data X1 and X2.
        """

        pairwise_distances = np.array(cdist(X1, X2, 'euclidean'))
        K = np.exp(-self.gamma * pairwise_distances**2)
        return np.array(K)