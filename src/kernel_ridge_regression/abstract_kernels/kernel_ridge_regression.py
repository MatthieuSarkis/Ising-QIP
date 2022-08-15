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
This module implements the abstract base class for kernel ridge regression.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve
from typing import Dict, Optional, Callable
from tqdm import tqdm

class KernelRidgeRegression(ABC):
    r"""Base class for Kernel Ridge Regression.
    This class implements three different fitting method:
    1) Analytic solution using Choleski decomposition
    2) Gradient descent
    3) AutoKer method: fits both the parameters AND the hyperparameters using gradient descent.
    """

    @abstractmethod
    def __init__(
        self,
        ridge_parameter: Optional[float] = 0.1,
        gamma: Optional[float] = None,
        *args,
        **kwargs
    ) -> None:
        r"""
        Initialize the Kernel Ridge Regression.
        Args:
            ridge_parameter: The ridge parameter.
            gamma: The gamma parameter.
        """

        super().__init__(*args, **kwargs)

        self.name: Optional[str] = None

        self.dataset: dict = {}
        self.hash_gram: Dict[int, np.ndarray] = {}
        self.hash_distances_squared: Dict[int, np.ndarray] = {}

        # Model-related hyperparameters
        self._ridge_parameter: Optional[float] = ridge_parameter
        self._gamma: Optional[float] = gamma

        # Parameters of the model
        self.alpha: Optional[np.ndarray] = None

    @property
    def ridge_parameter(self) -> float:
        return self._ridge_parameter

    @ridge_parameter.setter
    def ridge_parameter(self, ridge_parameter):
        self._ridge_parameter = ridge_parameter

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):

        self._gamma = gamma

    def fit_choleski(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        r"""Method to fit the parameters of the model using analytic expression.
        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The training labels associated to the trainning data.
            kernel (np.ndarray): The kernel function. Must be defined in the Child class.
        """

        # Get parameters
        self.dataset['X'] = X
        self.dataset['y'] = y

        # Check existence of the Gram matrix of the dataset X in the self.hash,
        # compute it if not already in self.hash_gram
        key = str(hash((X.tobytes(), X.tobytes(), self.gamma)))
        if not key in self.hash_gram:
            self.hash_gram[key] = self.kernel(X, X)
        K = self.hash_gram[key]

        # Analytic solution for the parameters of the model (the loss function is just quadratic in alpha)
        try:
            self.alpha = solve(K + self._ridge_parameter * np.eye(X.shape[0]), y, assume_a='pos')
        except (np.linalg.LinAlgError, ValueError, np.lina):
            return

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        number_epochs: int = 50000,
        learning_rate: float = 1e-5,
    ) -> None:
        r"""Method to fit the parameters of the model using gradient descent. If the dataset is small enough, the method fit_choleski is prefered.
        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The training labels associated to the trainning data.
            number_epochs (int): The number of epochs.
            learning_rate (float): The learning rate.
        """

        # Get parameters
        self.dataset['X'] = X
        self.dataset['y'] = y

        # Check existence of the Gram matrix of the dataset X in the self.hash,
        # compute it if not already in self.hash
        key = hash((X.tobytes(), X.tobytes(), self.gamma))
        if not key in self.hash_gram:
            self.hash_gram[key] = self.kernel(X, X)
        K = self.hash_gram[key]

        # Initialization of the parameters
        self.alpha = np.zeros(X.shape[0])

        # Entering in the training loop
        for _ in range(number_epochs):
            y_predicted = K.dot(self.alpha)

            # Kernelized loss function with Ridge regularization
            grad_alpha = 2 * K.dot(y_predicted - y) + 2 * self.ridge_parameter * y_predicted

            # Updating the parameters
            self.alpha -= learning_rate * grad_alpha

    def fit_autoker(
        self,
        X: np.ndarray,
        y: np.ndarray,
        number_epochs: int = 100,
        learning_rate_alpha: float = 1e-4,
        learning_rate_gamma: float = 1e-5,
        learning_rate_ridge_parameter: float = 1e-5,
    ) -> None:
        """Method to fit the parameters of the model using gradient descent.
        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The training labels associated to the trainning data.
            number_epochs (int): The number of epochs.
        """

        # Get parameters
        self.dataset['X'] = X

        # What we mean by 'distances_squared' here depends on the specific kernel we are looking at.
        # In the classical case it is just the matrix of Euclidean distances squared ||x_1 - x_2||^2.
        # In the quantum case it is ||\rho(x_1) - \rho(x_2)||_F^2 = 2 (1-|<psi_1|psi_2>|^2).
        key = str(hash((X.tobytes(), X.tobytes())))
        if not key in self.hash_distances_squared:
            self.hash_distances_squared[key] = self.distances_squared(X, X)
        distances_squared = self.hash_distances_squared[key]

        # Initialization of the parameters
        self.alpha = np.zeros(X.shape[0])
        self.ridge_parameter = 1.0
        self.gamma = 1.0 / (X.var() * X.shape[1])

        # Entering in the training loop
        for i in tqdm(range(number_epochs)):

            print(i)

            K = np.exp(-self.gamma * distances_squared)
            dK = -distances_squared * K

            y_predicted = K @ self.alpha

            grad_alpha = 2 * K @ (y_predicted - y) + 2 * self.ridge_parameter * y_predicted
            grad_gamma = (2 * (y_predicted - y) + self.ridge_parameter * self.alpha).T @ dK @ self.alpha
            grad_ridge_parameter = self.alpha.T @ K @ self.alpha

            self.alpha -= learning_rate_alpha * grad_alpha
            self.gamma -= learning_rate_gamma * grad_gamma
            self.ridge_parameter -= learning_rate_ridge_parameter * grad_ridge_parameter

            print(self.gamma)

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        r""" Method to predict the labels of given data.
        Args:
            X (np.ndarray): The data to predict the labels.
        Returns:
            (np.ndarray): The predicted labels.
        """

        key = hash((self.dataset['X'].tobytes(), X.tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = self.kernel(self.dataset['X'], X)
        K = self.hash_gram[key]

        y_predicted = np.sum(self.alpha.reshape(-1, 1) * K, axis=0)
        return y_predicted

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> float:
        r"""Method to evaluate the model with a given loss function.
        Args:
            X (np.ndarray): The data to predict the labels.
            y (np.ndarray): The labels associated to the data.
            loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): The loss function. Must be defined in the Child class.
        Returns:
            (float): The loss value.
        """

        y_predicted = self.predict(X)
        return float(loss(y, y_predicted))

    @abstractmethod
    def kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        r"""Method to compute the Gram matrix corresponding to the two batches of data X1 and X2.
        To be implemented by the children classes.
        Args:
            X1 (np.ndarray): First batch of data.
            X2 (np.ndarray): Second batch of data.
        """

        raise NotImplementedError

    @abstractmethod
    def distances_squared(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        r""" Compute the matrix of squared distances.
        Args:
            X1 (np.ndarray): First batch of 1-D array data vector.
            X2 (np.ndarray): Second batch of 1-D array data vector.
        Returns:
            (np.ndarray): Distances squared matrix associated to the two batches of data X1 and X2.
        """

        raise NotImplementedError