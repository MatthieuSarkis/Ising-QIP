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

from typing import Callable

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cdist

class AutoKer():

    def __init__(
        self,
        learning_rate_alpha: float = 1e-4,
        learning_rate_sigma: float = 1e-5,
        learning_rate_ridge_parameter: float = 1e-5
    ) -> None:

        super().__init__()

        self.name = 'gaussianAutoKer'
        self.dataset: dict = {}
        self.distances_squared: np.ndarray = None

        # learning rates
        self.learning_rate_alpha = learning_rate_alpha
        self.learning_rate_sigma = learning_rate_sigma
        self.learning_rate_ridge_parameter = learning_rate_ridge_parameter

        # Parameters of the model
        self.alpha = None
        self.ridge_parameter = None
        self.sigma = None

    def fit(
        self,
        X: ndarray,
        y: ndarray,
        number_epochs: int = 100
    ) -> None:
        """Method to fit the parameters of the model using gradient descent.
        Args:
            X: The training data.
            y: The training labels associated to the trainning data.
            number_epochs: The number of epochs.
        """

        # Get parameters
        self.dataset['X'] = X
        self.dataset['y'] = y
        self.number_epochs = number_epochs

        distances_squared = self.pairwise_distances_squared(X, X)

        # Initialization of the parameters
        self.alpha = np.zeros(X.shape[0])
        self.ridge_parameter = 0.1
        self.sigma = 1.0

        # Entering in the training loop
        for _ in range(self.number_epochs):

            K = np.exp(-distances_squared / (2 * self.sigma**2 + 1e-8))
            dK = distances_squared * K / (self.sigma**3 + 1e-8)

            y_predicted = K @ self.alpha

            grad_alpha = 2 * K @ (y_predicted - y) + 2 * self.ridge_parameter * y_predicted
            grad_sigma = (2 * (y_predicted - y) + self.ridge_parameter * self.alpha).T @ dK @ self.alpha
            grad_ridge_parameter = self.alpha.T @ K @ self.alpha

            self.alpha -= self.learning_rate_alpha * grad_alpha
            self.sigma -= self.learning_rate_sigma * grad_sigma
            self.ridge_parameter -= self.learning_rate_ridge_parameter * grad_ridge_parameter

    def predict(
        self,
        X: ndarray,
    ) -> ndarray:
        """ Method to predict the labels of given data.
        Args:
            X: The data to predict the labels.
        Returns:
            The predicted labels.
        """

        K = np.exp(-self.pairwise_distances_squared(self.dataset['X'], X) / (2 * self.sigma**2))
        y_predicted = np.sum(self.alpha.reshape(-1, 1) * K, axis=0)

        return y_predicted

    def evaluate(
        self,
        X: ndarray,
        y: ndarray,
        loss: Callable[[ndarray, ndarray], ndarray],
    ) -> float:
        """Method to evaluate the model with a given loss function.
        Args:
            X: The data to predict the labels.
            y: The labels associated to the data.
            loss: The loss function. Must be defined in the Child class.
        Returns:
            The loss value.
        """

        y_predicted = self.predict(X)
        return float(loss(y, y_predicted))

    @staticmethod
    def pairwise_distances_squared(
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        r""" Compute the Gaussian Kernel.
        Args:
            x1: A 1-D array data vector.
            x2: A 1-D array data vector.
            sigma: The standard deviation of the gaussian function.
        Returns:
            Distance matrix squared.
        """

        return np.array(cdist(x1, x2, 'euclidean'))**2