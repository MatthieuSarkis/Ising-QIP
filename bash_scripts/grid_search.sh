#!/bin/bash
# Purpose: Run a grid search with qpu over the hyperparameters
# Usage: ./bash_scripts/grid_search.sh
# Comment: The grid is defined by the cartesian product of
# two lists of hyperparameters: RIDGE_PARAMETER and GAMMA,
# that can be found and modified at will in
# 'src/main/main_grid_search.py'

IMAGE_SIZE=4
DATASET_SIZE=20
#REGRESSOR='gaussian'
REGRESSOR='quantum'
MEMORY_BOUND=5

BACKEND_TYPE="simulator"
BACKEND_NAME="statevector_simulator"
#BACKEND_NAME="qasm_simulator"
SEED=42
SHOTS=1024
HUB="ibm-q"
GROUP="open"
PROJECT="main"
JOB_NAME="ising_qip"

python src/main/main_grid_search.py \
    --image_size $IMAGE_SIZE \
    --dataset_size $DATASET_SIZE \
    --regressor $REGRESSOR \
    --memory_bound $MEMORY_BOUND \
    --backend_type $BACKEND_TYPE \
    --backend_name $BACKEND_NAME \
    --no-mitigate \
    --no-use_ancilla \
    --seed $SEED \
    --shots $SHOTS \
    --hub $HUB \
    --group $GROUP \
    --project $PROJECT \
    --job_name $JOB_NAME