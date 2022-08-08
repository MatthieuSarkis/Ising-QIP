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
Implementation of grid search algorithm for the kernel ridge regression.
"""


import os
import itertools
import json
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, List

from src.utils.loss_functions import rmae
from src.kernel_ridge_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression


class Grid():

    def __init__(
        self,
        regressor: KernelRidgeRegression,
        ridge_parameter: List[float],
        gamma: List[float],
        save_directory: str,
        criterion: Callable[[np.ndarray, np.ndarray], np.ndarray] = rmae,
    ) -> None:
        r"""Constructor for the Grid class.
        Args:
            regressor (KernelRidgeRegression): A regressor whose hyperparameters one is optimizing over.
            ridge_parameter (List[float]): List of values for the ridge parameter.
            gamma (List[float]): List of values for the width parameter gamma.
            save_directory (str): directory where to save the optimized hyperparameters.
            criterion (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function used.
        """

        super(Grid, self).__init__()
        self.name = 'grid'
        self.regressor = regressor
        self.criterion = criterion
        self.save_directory = save_directory

        self.hyperparameters_grid = self._prepare_grid(
            ridge_parameter=ridge_parameter,
            gamma=gamma
        )

        self._theta: Dict[str, float] = None
        self._dataset: Dict[str, np.ndarray] = None
        self.J: float = None
        self.best_hyperparameters_json: str = None

    def _initialize(
        self,
    ) -> None:

        self._dataset = {
            'X_train': None,
            'y_train': None,
            'X_val': None,
            'y_val': None
        }

        self._theta = {
            'ridge_parameter': None,
            'gamma': None
        }

        self.J: float = float('inf')

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = False
    ) -> None:
        r"""Method performing the grid search
        Args:
            X_train (np.ndarray): batch of train images.
            y_train (np.ndarray): batch of train labels.
            X_val (np.ndarray): batch of validation images.
            y_val (np.ndarray): batch of train labels.
            verbose (bool): output some progess message.
        """

        self._initialize()

        self.dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }

        for theta in tqdm(self.hyperparameters_grid):

            self._one_step(theta=theta, verbose=verbose)
            self._dump_hyperparameters(save_directory=self.save_directory)

    def _one_step(
        self,
        theta: Dict[str, float],
        verbose: bool = False
    ) -> None:
        r"""Fits the regressor for the given values of the hyperparameters and check
        if there is improvement.
        Args:
            theta (Dict[str, float]): set of the values of the hyperparameters.
            verbose (bool): output some progess message.
        """

        X_train = self.dataset['X_train']
        y_train = self.dataset['y_train']
        X_val = self.dataset['X_val']
        y_val = self.dataset['y_val']

        self.regressor.ridge_parameter = theta['ridge_parameter']
        self.regressor.gamma = theta['gamma']

        try:
            self.regressor.fit_choleski(X_train, y_train)
        except:
            return

        J = self.regressor.evaluate(X_val, y_val, loss=self.criterion)

        if J < self.J:
            self.J = J
            self.theta = theta
            if verbose:
                print("rmae = {:.2f}%".format(100 * self.J))

    def _dump_hyperparameters(
        self,
        save_directory: str = './saved_models',
    ) -> None:
        r"""Dumps the optimal values of the hyperparameters to a file.
        Args:
            save_directory (str): directory where to save the optimized hyperparameters.
        """

        assert self.theta is not None, "Call the fit method first!"

        d = self.theta.copy()

        d['train_size'] = self.dataset['X_train'].shape[0]
        d['val_size'] = self.dataset['X_val'].shape[0]
        d['regressor_name'] = self.regressor.name
        d[self.criterion.__name__] = self.J
        if hasattr(self.regressor, 'mitigate'):
            d['mitigate'] = self.regressor.mitigate

        self.best_hyperparameters_json = os.path.join(save_directory, 'hyperparams.json')

        with open(self.best_hyperparameters_json, 'w') as f:
            json.dump(d, f, indent=4)

    def evaluate_best_model(self) -> float:
        r"""Returns the validation loss corresponding to the best regressor."""

        assert self.J is not None, "Call the fit method first!"
        return self.J

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(
        self,
        dataset: Dict[str, np.ndarray]
    ) -> None:

        self._dataset['X_train'] = dataset['X_train']
        self._dataset['y_train'] = dataset['y_train']
        self._dataset['X_val'] = dataset['X_val']
        self._dataset['y_val'] = dataset['y_val']

    @property
    def theta(self) -> Dict[str, float]:
        return self._theta

    @theta.setter
    def theta(
        self,
        theta: Dict[str, float]
    ) -> None:

        self._theta['ridge_parameter'] = theta['ridge_parameter']
        self._theta['gamma'] = theta['gamma']

    @staticmethod
    def _prepare_grid(
        ridge_parameter: List[float],
        gamma: List[float],
    ) -> List[Dict[str, float]]:
        r"""Prepares the grid to search over.
        Args:
            ridge_parameter (List[float]): List of values for the ridge parameter.
            gamma (List[float]): List of values for the width parameter gamma.
        Returns:
            (List[Dict[str, float]]): The hyperparameters grid.
        """

        hyperparams_grid: List[Dict[str, float]] = []

        for pair in itertools.product(ridge_parameter, gamma):

            hyperparams_grid.append({
                'ridge_parameter': pair[0],
                'gamma': pair[1]
            })

        return hyperparams_grid