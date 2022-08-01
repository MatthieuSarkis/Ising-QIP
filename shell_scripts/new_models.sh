#DATASET="qm7x"
#DATA_PATH_X="./data/qm7x/X_y/full_1conf_noH_coulomb_X.npy"
#DATA_PATH_Y="./data/qm7x/X_y/full_1conf_noH_coulomb_y.npy"

DATASET="qm7x"
DATA_PATH_X="./data/qm7x/X_y/full_1conf_noH_coulomb_X.npy"
DATA_PATH_Y="./data/qm7x/X_y/full_1conf_noH_coulomb_y.npy"

REGRESSOR="quantum_zz_vectorized"

#for PROPERTY in eAT HLgap
#do
#	python src/main/main_grid_search_new_kernels.py \
#        --dataset $DATASET \
#        --data_path_X $DATA_PATH_X \
#        --data_path_y $DATA_PATH_Y \
#        --dataset_size 3500 \
#        --property $PROPERTY \
#        --regressor gaussian
#done
#
#for PROPERTY in eAT HLgap
#do
#    for N_NON_TRACED_OUT_QUBIT in 1 2 3 4 all
#    do
#        for KERNEL_TYPE in classical_copy quantum_gaussian pqk linear
#	    do
#            python src/main/main_grid_search_new_kernels.py \
#                --dataset $DATASET \
#                --data_path_X $DATA_PATH_X \
#                --data_path_y $DATA_PATH_Y \
#                --dataset_size 3500 \
#                --property $PROPERTY \
#                --regressor $REGRESSOR \
#                --n_non_traced_out_qubits $N_NON_TRACED_OUT_QUBIT \
#                --kernel_type $KERNEL_TYPE
#        done
#    done
#done

#for PROPERTY in HLgap
#do
#    for N_NON_TRACED_OUT_QUBIT in 1 2 3 4 5 all
#    do
#        for KERNEL_TYPE in quantum_gaussian
#	    do
#            for ENTANGLEMENT in linear circular full
#            do
#                python src/main/main_grid_search_new_kernels.py \
#                    --dataset $DATASET \
#                    --data_path_X $DATA_PATH_X \
#                    --data_path_y $DATA_PATH_Y \
#                    --dataset_size 3500 \
#                    --property $PROPERTY \
#                    --regressor $REGRESSOR \
#                    --n_non_traced_out_qubits $N_NON_TRACED_OUT_QUBIT \
#                    --kernel_type $KERNEL_TYPE \
#                    --entanglement $ENTANGLEMENT \
#                    --reps 1
#            done
#        done
#    done
#done

#for PROPERTY in HLgap
#do
#    for N_NON_TRACED_OUT_QUBIT in 1 2 3 4 5 all
#    do
#        for KERNEL_TYPE in quantum_gaussian
#	    do
#            for ENTANGLEMENT in linear circular full
#            do
#                python src/main/main_simple_fit.py \
#                    --dataset $DATASET \
#                    --data_path_X $DATA_PATH_X \
#                    --data_path_y $DATA_PATH_Y \
#                    --dataset_size 15000 \
#                    --property $PROPERTY \
#                    --regressor $REGRESSOR \
#                    --n_non_traced_out_qubits $N_NON_TRACED_OUT_QUBIT \
#                    --kernel_type $KERNEL_TYPE \
#                    --entanglement $ENTANGLEMENT \
#                    --reps 1
#            done
#        done
#    done
#done

for PROPERTY in HLgap
do
	python src/main/main_grid_search_new_kernels.py \
        --dataset $DATASET \
        --data_path_X $DATA_PATH_X \
        --data_path_y $DATA_PATH_Y \
        --dataset_size 3500 \
        --property $PROPERTY \
        --regressor gaussian
done

for PROPERTY in HLgap
do
    python src/main/main_simple_fit.py \
        --dataset $DATASET \
        --data_path_X $DATA_PATH_X \
        --data_path_y $DATA_PATH_Y \
        --dataset_size 15000 \
        --property $PROPERTY \
        --regressor gaussian
done

#do
#for PROPERTY in HLgap
#    for N_NON_TRACED_OUT_QUBIT in 1 2 3 4 5 all
#    do
#        for KERNEL_TYPE in quantum_gaussian
#	    do
#            for ENTANGLEMENT in linear circular full
#            do
#                python src/main/main_simple_fit.py \
#                    --dataset $DATASET \
#                    --data_path_X ./data/qm7x/X_y/full_2conf_noH_coulomb_X.npy \
#                    --data_path_y ./data/qm7x/X_y/full_2conf_noH_coulomb_y.npy \
#                    --dataset_size 15000 \
#                    --property $PROPERTY \
#                    --regressor $REGRESSOR \
#                    --n_non_traced_out_qubits $N_NON_TRACED_OUT_QUBIT \
#                    --kernel_type $KERNEL_TYPE \
#                    --entanglement $ENTANGLEMENT \
#                    --reps 1
#            done
#        done
#    done
#done

#
#for PROPERTY in eAT HLgap
#do
#    for N_NON_TRACED_OUT_QUBIT in 1 2 3 4 all
#    do
#        for KERNEL_TYPE in classical_copy quantum_gaussian pqk linear
#	    do
#            python src/main/main_benchmark.py \
#                --dataset $DATASET \
#                --data_path_X $DATA_PATH_X \
#                --data_path_y $DATA_PATH_Y \
#                --dataset_size 5000 \
#                --property $PROPERTY \
#                --regressor $REGRESSOR \
#                --n_non_traced_out_qubits $N_NON_TRACED_OUT_QUBIT \
#                --kernel_type $KERNEL_TYPE \
#                --entanglement full \
#                --step 50
#        done
#    done
#done

#python src/main/main_benchmark.py \
#    --dataset $DATASET \
#    --data_path_X $DATA_PATH_X \
#    --data_path_y $DATA_PATH_Y \
#    --dataset_size 15000 \
#    --property eAT \
#    --regressor gaussian \
#    --step 200
#
#python src/main/main_benchmark.py \
#    --dataset $DATASET \
#    --data_path_X $DATA_PATH_X \
#    --data_path_y $DATA_PATH_Y \
#    --dataset_size 15000 \
#    --property eAT \
#    --regressor $REGRESSOR \
#    --n_non_traced_out_qubits 4 \
#    --kernel_type quantum_gaussian \
#    --step 200

#for PROPERTY in eAT HLgap
#do
#    for N_NON_TRACED_OUT_QUBIT in 1 2 3 4 all
#    do
#        for KERNEL_TYPE in classical_copy quantum_gaussian pqk linear
#	    do
#            python src/main/main_simple_fit.py \
#                --dataset $DATASET \
#                --data_path_X $DATA_PATH_X \
#                --data_path_y $DATA_PATH_Y \
#                --dataset_size 15000 \
#                --property $PROPERTY \
#                --regressor $REGRESSOR \
#                --n_non_traced_out_qubits $N_NON_TRACED_OUT_QUBIT \
#                --kernel_type $KERNEL_TYPE
#        done
#    done
#done

#for PROPERTY in eAT HLgap
#do
#    python src/main/main_simple_fit.py \
#        --dataset $DATASET \
#        --data_path_X $DATA_PATH_X \
#        --data_path_y $DATA_PATH_Y \
#        --dataset_size 15000 \
#        --property $PROPERTY \
#        --regressor gaussian
#done

#python src/main/main_simple_fit.py \
#    --dataset $DATASET \
#    --data_path_X $DATA_PATH_X \
#    --data_path_y $DATA_PATH_Y \
#    --dataset_size 15000 \
#    --property eAT \
#    --regressor $REGRESSOR \
#    --n_non_traced_out_qubits 2 \
#    --kernel_type classical_copy

#for PROPERTY in eElectronic HLgap
#do
#    python src/main/main_quantum_criteria.py \
#        --data_path_X $DATA_PATH_X \
#        --data_path_y $DATA_PATH_Y \
#        --dataset_size 5000 \
#        --property $PROPERTY \
#        --regressor quantum_zz_vectorized \
#        --n_non_traced_out_qubits all \
#        --kernel_type quantum_gaussian\
#        --step 50
#done