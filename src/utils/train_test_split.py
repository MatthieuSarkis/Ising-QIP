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

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.25,
) -> tuple:

    dataset_size = y.shape[0]

    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    split_idx = int(dataset_size * (1 - validation_fraction))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    return X_train, y_train, X_val, y_val