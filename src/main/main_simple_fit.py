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

from argparse import ArgumentParser
import json
import numpy as np
import os
from typing import Tuple

from src.data_factory.pca import PCA
from src.data_factory.utils import Normalizer
from src.data_factory.utils import rmae, train_test_split
from src.quantum_regression.abstract_kernels.kernel_ridge_regression import KernelRidgeRegression
from src.quantum_regression.kernels.gaussian import GAUSSIAN
from src.quantum_regression.kernels.quantum_zz_vectorized import QUANTUM_ZZ_VECTORIZED
from src.quantum_regression.kernels.quantum_raw_vectorized import QUANTUM_RAW_VECTORIZED
from src.benchmark.benchmark import Benchmark

def load_data(
    dataset: str = 'qm7x',
    data_path_X: str = './data/qm7x/X_y/full_1conf_noH_coulomb_X.npy',
    data_path_y: str = './data/qm7x/X_y/full_1conf_noH_coulomb_y.npy',
    dataset_size: int = 1000,
    n_atoms: int = 10000,
    property: str = 'eElectronic',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    with open(data_path_X, 'rb') as f:
        X = np.load(f)
    with open(data_path_y, 'rb') as f:
        y = np.load(f)

    X = X[y[:, -1] <= n_atoms][:, :n_atoms*(n_atoms+1)//2]
    y = y[y[:, -1] <= n_atoms]

    if dataset == 'qm7x':
        if property == 'eElectronic':
            y = y[:, 10] - y[:, 16]
        elif property == 'eAT':
            y = y[:, 12]
        elif property == 'HLgap':
            y = y[:, 27]
        elif property == 'DIP':
            y = y[:, 28]
        elif property == 'mPOL':
            y = y[:, 42]

    elif dataset == 'qm9':
        if property == 'eAT':
            y = y[:, 3] # U0
        elif property == 'HLgap':
            y = y[:, 2] # gap

    idx = np.random.permutation(X.shape[0])

    X = X[idx]
    y = y[idx]
    X = X[:dataset_size]
    y = y[:dataset_size]

    return train_test_split(X, y)

def instantiate_regressor(
    args,
    sigma: float,
    ridge_parameter: float
) -> KernelRidgeRegression:

    if args.regressor == 'gaussian':
        regressor = GAUSSIAN(sigma=sigma, ridge_parameter=ridge_parameter)

    elif args.regressor == 'quantum_raw_vectorized':
        regressor = QUANTUM_RAW_VECTORIZED(
            sigma=sigma,
            ridge_parameter=ridge_parameter,
            n_non_traced_out_qubits=args.n_non_traced_out_qubits,
            kernel_type=args.kernel_type
        )

    elif args.regressor == 'quantum_zz_vectorized':
        regressor = QUANTUM_ZZ_VECTORIZED(
            sigma=sigma,
            ridge_parameter=ridge_parameter,
            n_non_traced_out_qubits=args.n_non_traced_out_qubits,
            kernel_type=args.kernel_type,
            entanglement=args.entanglement,
            reps=args.reps
        )

    return regressor

def main(args) -> None:

    # Create the log directory tree
    if args.regressor == 'gaussian':
        save_directory = os.path.join(
            "./saved_models",
            "grid",
            "dataset="+args.dataset,
            'property={}'.format(args.property),
            'regressor={}'.format(args.regressor)
        )

    elif args.regressor == 'quantum_raw_vectorized':
        save_directory = os.path.join(
            "./saved_models",
            "grid",
            "dataset="+args.dataset,
            'property={}'.format(args.property),
            'regressor={}'.format(args.regressor),
            'kernel_type={}'.format(args.kernel_type),
            'non_traced_out_qubits={}'.format(args.n_non_traced_out_qubits)
        )

    elif args.regressor == 'quantum_zz_vectorized':
        save_directory = os.path.join(
            "./saved_models",
            "grid",
            "dataset="+args.dataset,
            'property={}'.format(args.property),
            'regressor={}'.format(args.regressor),
            'kernel_type={}'.format(args.kernel_type),
            'entanglement={}'.format(args.entanglement),
            'non_traced_out_qubits={}'.format(args.n_non_traced_out_qubits)
        )

    os.makedirs(save_directory, exist_ok=True)

    # Load the data
    X_train, y_train, X_val, y_val = load_data(
        dataset=args.dataset,
        data_path_X=args.data_path_X,
        data_path_y=args.data_path_y,
        dataset_size=args.dataset_size,
        property=args.property
    )

    if args.regressor == 'quantum_zz_vectorized':

        # the number of qubits scales like the classical feature space dimension
        # so one should reduce dimensionality first, using a PCA for instance
        pca = PCA(n_features=7)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)

        # the data should be normalized to sit in the [0, 2\pi] interval.
        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train)
        X_val = normalizer.transform(X_val)

    with open(os.path.join(save_directory, 'hyperparams.json')) as f:
        hyperparams = json.load(f)

    regressor = instantiate_regressor(
        args=args,
        sigma=hyperparams['sigma'],
        ridge_parameter=hyperparams['ridge_parameter']
    )

    print(save_directory)

    regressor.fit_choleski(
        X=X_train,
        y=y_train
    )

    loss = regressor.evaluate(
        X=X_val,
        y=y_val,
        loss=rmae
    )

    with open(os.path.join(save_directory, 'single_loss.txt'), 'w') as f:
        f.write("Train dataset size = {}\n".format(X_train.shape[0]))
        f.write("Validation dataset size = {}\n".format(X_val.shape[0]))
        f.write("Validation loss (relative MAE in %) = {:.2f}".format(loss * 100))

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default='qm7x', choices=['qm7x', 'qm9'])
    parser.add_argument("--data_path_X", type=str, default='./data/qm7x/X_y/full_1conf_noH_coulomb_X.npy')
    parser.add_argument("--data_path_y", type=str, default='./data/qm7x/X_y/full_1conf_noH_coulomb_y.npy')
    parser.add_argument("--property", type=str, default='eElectronic', choices=['eElectronic', 'eAT', 'HLgap', 'DIP', 'mPOL'])
    parser.add_argument("--regressor", type=str, default='gaussian', choices=['gaussian', 'quantum_raw_vectorized', 'quantum_zz_vectorized'])
    parser.add_argument("--dataset_size", type=int, default=10000)

    # arguments for common to all quantum kernels
    parser.add_argument("--kernel_type", type=str, default='classical_copy', choices=['classical_copy', 'quantum_gaussian_sqrt', 'quantum_gaussian', 'pqk', 'linear'])
    parser.add_argument("--n_non_traced_out_qubits", default='all') # [1, 2, 3, ..., 'all']

    # arguments for the zz-feature map
    parser.add_argument("--entanglement", type=str, default='full', choices=['linear', 'circular', 'full'])
    parser.add_argument("--reps", type=int, default=1)

    args = parser.parse_args()

    if args.n_non_traced_out_qubits is not None and args.n_non_traced_out_qubits.isdigit():
        args.n_non_traced_out_qubits = int(args.n_non_traced_out_qubits)

    main(args=args)