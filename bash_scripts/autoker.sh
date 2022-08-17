#!/bin/bash
# Purpose: Run a grid search with qpu over the hyperparameters
# Usage: ./bash_scripts/grid_search.sh
# Comment: The grid is defined by the cartesian product of
# two lists of hyperparameters: RIDGE_PARAMETER and GAMMA,
# that can be found and modified at will in
# 'src/main/main_grid_search.py'

IMAGE_SIZE=16
DATASET_SIZE=10
REGRESSOR='gaussian'
#REGRESSOR='quantum'
MEMORY_BOUND=3

BACKEND_TYPE="simulator"
BACKEND_NAME="statevector_simulator"
SEED=42
SHOTS=1024
HUB="ibm-q"
GROUP="open"
PROJECT="main"
JOB_NAME="ising_qip"

python src/main/main_autoker.py \
    --image_size $IMAGE_SIZE \
    --dataset_size $DATASET_SIZE \
    --regressor $REGRESSOR \
    --learning_rate_alpha 1e-4 \
    --learning_rate_gamma 1e-7 \
    --learning_rate_ridge_parameter 1e-5 \
    --number_epochs 1000 \
    --memory_bound $MEMORY_BOUND \
    --backend_type $BACKEND_TYPE \
    --backend_name $BACKEND_NAME \
    --no-mitigate \
    --seed $SEED \
    --shots $SHOTS \
    --hub $HUB \
    --group $GROUP \
    --project $PROJECT \
    --job_name $JOB_NAME