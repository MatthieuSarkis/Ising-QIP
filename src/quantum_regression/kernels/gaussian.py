# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis) and Adel Sohbi (https://github.com/adelshb).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Implementation of the Gaussian Kernel Ridge Regression. """

import numpy as np
from scipy.spatial.distance import cdist

from src.kernel_ridge_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression

class GAUSSIAN(KernelRidgeRegression):
    """ Gaussian Kernel for Kernel Ridge Regression. """

    def __init__(
        self,
        *args, 
        **kwargs,
    ) -> None:
        r""" Initialize the Gaussian Kernel Ridge Regression from parent's init. """
        
        super(GAUSSIAN, self).__init__(*args, **kwargs)  
        self.name = 'gaussian'
   
    def kernel(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the Gaussian Kernel. 
        Args:
            x1: A 1-D array data vector.
            x2: A 1-D array data vector.
            sigma: The standard deviation of the gaussian function.
        Returns:
            Th kernel matrix.
        """
         
        pairwise_distances = np.array(cdist(x1, x2, 'euclidean'))
        K = np.exp(-pairwise_distances**2 / (2 * self.sigma**2))
        return np.array(K)