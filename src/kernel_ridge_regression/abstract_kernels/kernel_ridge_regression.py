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

"""
This module implements the abstract base class for kernel ridge regression.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
from numpy import ndarray
from scipy.linalg import solve
from typing import Dict, Optional

class KernelRidgeRegression(ABC):
    """
    Base class for Kernel Ridge Regression.
    This method should initialize the module and use an exception if a component of the module is available.
    """
    @abstractmethod
    def __init__(
        self,
        ridge_parameter: Optional[float] = 0.1,
        sigma: Optional[float] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Initialize the Kernel Ridge Regression.
        Args:
            ridge_parameter: The ridge parameter.
            sigma: The sigma parameter.
        """

        super().__init__(*args, **kwargs)

        self.name: Optional[str] = None

        self.dataset: dict = {}
        self.hash_gram: Dict[int, ndarray] = {}

        # Model-related hyperparameters
        self._ridge_parameter = ridge_parameter
        self._sigma = sigma

        # Parameters of the model
        self.alpha: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None

    @property
    def ridge_parameter(self) -> float:
        return self._ridge_parameter

    @ridge_parameter.setter
    def ridge_parameter(self, ridge_parameter):
        self._ridge_parameter = ridge_parameter

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):

        # setting a new value of sigma means that we should recompute Gram matrices
        self.hash_gram = {}
        self._sigma = sigma

    def fit_choleski(
        self,
        X: ndarray,
        y: ndarray,
    ) -> None:
        """Method to fit the parameters of the model using analytic expression.
        Args:
            X: The training data.
            y: The training labels associated to the trainning data.
            kernel: The kernel function. Must be defined in the Child class.
        """

        # Get parameters
        self.dataset['X'] = X
        self.dataset['y'] = y

        # Check existence of the Gram matrix of the dataset X in the self.hash,
        # compute it if not already in self.hash
        key = hash((X.tobytes(), X.tobytes()))
        #print(self.hash_gram)
        if not key in self.hash_gram:
            self.hash_gram[key] = self.kernel(X, X)
        self.K = self.hash_gram[key]

        # Analytic solution for the parameters of the model (the loss function is just quadratic in alpha)
        try:
            self.alpha = solve(self.K + self._ridge_parameter * np.eye(X.shape[0]), y, assume_a='pos')
        except (np.linalg.LinAlgError, ValueError, np.lina):
            return

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        number_epochs: int = 50000,
        learning_rate: float = 1e-5,
    ) -> None:
        """Method to fit the parameters of the model using gradient descent. If the dataset is small enough, the method fit_choleski is prefered.
        Args:
            X: The training data.
            y: The training labels associated to the trainning data.
            kernel: The kernel function. Must be defined in the Child class.
            number_epochs: The number of epochs.
            learning_rate: The learning rate.
        """

        # Get parameters
        self.dataset['X'] = X
        self.dataset['y'] = y
        self.number_epochs = number_epochs
        self.learning_rate = learning_rate

        # Check existence of the Gram matrix of the dataset X in the self.hash,
        # compute it if not already in self.hash
        key = hash((X.tobytes(), X.tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = self.kernel(X, X)
        self.K = self.hash_gram[key]

        # Initialization of the parameters
        self.alpha = np.zeros(X.shape[0])

        # Entering in the training loop
        for _ in range(self.number_epochs):
            y_predicted = self.K.dot(self.alpha)

            # Kernelized loss function with Ridge regularization
            grad_alpha = 2 * self.K.dot(y_predicted - y) + 2 * self.ridge_parameter * y_predicted

            # Updating the parameters
            self.alpha -= self.learning_rate * grad_alpha

    def predict(
        self,
        X: ndarray,
    ) -> ndarray:
        """ Method to predict the labels of given data.
        Args:
            X: The data to predict the labels.
            kernel: The kernel function. Must be defined in the Child class.
        Returns:
            The predicted labels.
        """

        key = hash((self.dataset['X'].tobytes(), X.tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = self.kernel(self.dataset['X'], X)
        K = self.hash_gram[key]

        y_predicted = np.sum(self.alpha.reshape(-1, 1) * K, axis=0)
        return y_predicted

    def evaluate(
        self,
        X: ndarray,
        y: ndarray,
        loss: Callable[[ndarray, ndarray], ndarray]
    ) -> float:
        """Method to evaluate the model with a given loss function.
        Args:
            X (np.ndarray): The data to predict the labels.
            y (np.ndarray): The labels associated to the data.
            loss (Callable[[ndarray, ndarray], ndarray]): The loss function. Must be defined in the Child class.
        Returns:
            (float): The loss value.
        """

        y_predicted = self.predict(X)
        return float(loss(y, y_predicted))

    @abstractmethod
    def kernel(
        self,
        X1: ndarray,
        X2: ndarray
    ) -> ndarray:

        raise NotImplementedError