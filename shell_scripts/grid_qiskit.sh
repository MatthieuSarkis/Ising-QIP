#!/bin/bash
# Purpose: Run a grid search with qpu over the hyperparameters of 
# the three types of regressors
# Usage: ./shell_scripts/grid_search.sh
# Comment: The grid is defined by the cartesian product of 
# two lists of hyperparameters: RIDGE_PARAMETER and SIGMA, 
# that can be found and modified at will in 
# 'src/main/main_grid_qiskit.py'

python src/main/main_grid_qiskit.py \
    --regressor qiskit_raw \
    --dataset qm7x \
    --data_path_X ./data/qm7x/X_y/full_1conf_H_coulomb_X.npy \
    --data_path_y ./data/qm7x/X_y/full_1conf_H_coulomb_y.npy \
    --property eElectronic \
    --dataset_size 700 \
    --n_non_traced_out_qubits 1 \
    --backend_name aer_simulator \
    --mitigate False \
    --seed 42 \
    --shots 1024 \
    --backend_type simulator \
    --hub ibm-q \
    --group open \
    --project main \
    --job_name pqk