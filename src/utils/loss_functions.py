# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

# Loss functions
def rmae(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> np.ndarray:

    return np.mean(np.abs((y_true - y_predicted) /  y_true))

def mse(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> np.ndarray:

    return np.mean((y_true - y_predicted)**2)

def rmse(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> np.ndarray:

    return 1 / y_true.shape[0] * np.sqrt(np.sum((y_true - y_predicted)**2))

def percentage_mae(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> np.ndarray:

    return np.mean(np.abs(y_true - y_predicted) / (np.abs(y_true) + np.abs(y_predicted)))