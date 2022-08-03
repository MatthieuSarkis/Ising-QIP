IMAGE_SIZE=16
DATASET_SIZE=2
REGRESSOR='gaussian'

BACKEND_TYPE="simulator"
BACKEND_NAME="statevector_simulator"
MITIGATE=False
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
    --backend_name $BACKEND_NAME \
    --mitigate $MITIGATE \
    --seed $SEED \
    --shots $SHOTS \
    --hub $HUB \
    --group $GROUP \
    --project $PROJECT \
    --job_name $JOB_NAME