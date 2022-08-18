#!/bin/bash
# Purpose: Run a grid search with qpu over the hyperparameters
# Usage: ./bash_scripts/grid_search.sh
# Comment: The grid is defined by the cartesian product of
# two lists of hyperparameters: RIDGE_PARAMETER and GAMMA,
# that can be found and modified at will in
# 'src/main/main_grid_search.py'

IMAGE_SIZE=4
DATASET_SIZE=10
#REGRESSOR='gaussian'
REGRESSOR='quantum'
BACKEND_TYPE="simulator"
#BACKEND_TYPE="IBMQ"
#BACKEND_NAME="statevector_simulator"
BACKEND_NAME="qasm_simulator"
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
    --backend_type $BACKEND_TYPE \
    --no-mitigate \
    --use_ancilla \
    --parallelize \
    --seed $SEED \
    --shots $SHOTS \
    --hub $HUB \
    --group $GROUP \
    --project $PROJECT \
    --job_name $JOB_NAME \
    --backend_name $BACKEND_NAME