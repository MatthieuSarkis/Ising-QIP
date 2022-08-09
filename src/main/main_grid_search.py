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


from argparse import ArgumentParser
import os
import glob
import numpy as np
from scipy.stats import moment
import warnings
warnings.filterwarnings("ignore")
from typing import Tuple, List

from src.kernel_ridge_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression
from src.kernel_ridge_regression.kernels.classical_kernel import Gaussian
from src.kernel_ridge_regression.kernels.quantum_kernel import Quantum_Kernel
from src.kernel_ridge_regression.grid import Grid
from src.utils.train_test_split import train_test_split


def load_data(
    image_size: int = 16,
    dataset_size = 5000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    images = []
    labels = []

    for file_path in glob.glob('./data/ising/L={}/*'.format(image_size)):
        with open(file_path, 'rb') as f:
            X = np.frombuffer(buffer=f.read(), dtype=np.int8, offset=0).reshape(-1, image_size, image_size)
            temperature = float(file_path.split('=')[-1].split('.bin')[0])
            y = np.full(shape=(X.shape[0],), fill_value=temperature)

            images.append(X)
            labels.append(y)

    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)

    # Permuting the dataset
    idx = np.random.permutation(X.shape[0])
    X = X[idx][:dataset_size]
    y = y[idx][:dataset_size]

    # Splitting train set into train and validation set
    X_train, y_train, X_test, y_test = train_test_split(X, y, validation_fraction=0.2)

    return X_train, y_train, X_test, y_test

def make_grid(
    feature_dim: int,
    scale: float,
) -> Tuple[List[float], List[float]]:

    GAMMA = map(lambda x: x / (feature_dim * scale), [2.0**k for k in range(-5, 15)]) # cf. (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
    RIDGE_PARAMETER = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.1]

    return GAMMA, RIDGE_PARAMETER

def instantiate_regressor(args) -> KernelRidgeRegression:

    if args.regressor == 'gaussian':
        regressor = Gaussian()

    elif args.regressor == 'quantum':
        regressor = Quantum_Kernel(
            memory_bound=args.memory_bound,
            backend_type=args.backend_type,
            backend_name=args.backend_name,
            mitigate=args.mitigate,
            seed=args.seed,
            shots=args.shots,
            hub=args.hub,
            group=args.group,
            project=args.project,
            job_name=args.job_name,
        )

    return regressor

def main(args) -> None:

    # Create the log directory tree
    save_directory = os.path.join("./saved_models", "grid", 'regressor={}'.format(args.regressor))
    os.makedirs(save_directory, exist_ok=True)

    # Load the data
    X_train, y_train, X_val, y_val = load_data(image_size=args.image_size, dataset_size=args.dataset_size)
    # Flatten the image for the Gaussian kernel regressor
    if args.regressor == 'gaussian':
        X_train = X_train.reshape(-1, args.image_size**2)
        X_val = X_val.reshape(-1, args.image_size**2)

    # Instanciate the regressor and the grid
    GAMMA, RIDGE_PARAMETER = make_grid(feature_dim=X_train.shape[1], scale=moment(X_train, axis=None, moment=2 if args.regressor=='gaussian' else 4))
    regressor = instantiate_regressor(args=args)
    grid = Grid(regressor=regressor, ridge_parameter=RIDGE_PARAMETER, gamma=GAMMA, save_directory=save_directory)

    # Run the grid search
    grid.fit(X_train, y_train, X_val, y_val)

if __name__ == '__main__':

    parser = ArgumentParser()

    # Dataset parameters
    parser.add_argument("--image_size", type=int, default=16)
    parser.add_argument("--dataset_size", type=int, default=100)
    parser.add_argument("--regressor", type=str, default='gaussian', choices=['gaussian', 'quantum'])
    parser.add_argument("--memory_bound", type=int)

    # Qiskit parameters
    parser.add_argument("--backend_type", type=str, default="simulator")
    parser.add_argument("--backend_name", type=str, default="statevector_simulator")
    parser.add_argument('--mitigate', dest='mitigate', action='store_true')
    parser.add_argument('--no-mitigate', dest='mitigate', action='store_false')
    parser.set_defaults(set_mitigate=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--hub", type=str, default="ibm-q")
    parser.add_argument("--group", type=str, default="open")
    parser.add_argument("--project", type=str, default="main")
    parser.add_argument("--job_name", type=str, default="ising_qip")

    args = parser.parse_args()
    main(args=args)