#!/bin/bash
#SBATCH --job-name=ilu_scale
#SBATCH --time=02:00:00        # Increased time limit for larger matrices
#SBATCH --output=logs/ilu_scale_w%n_j%j.out  # %n = ntasks, %j = job ID

source .venv/bin/activate

TESTS_PER_CONFIG=20
WORKERS=$SLURM_NTASKS

echo "=== Starting Scaling Test ==="
echo "Workers (MPI Ranks): $WORKERS"
echo "Tests per config: $TESTS_PER_CONFIG"
echo "============================="

# Loop over matrix sizes
for N_SIZE in 10 100 1000; do
    
    # Loop over densities
    for DENSITY in 0.1 0.2 0.3; do
        
        echo ">> Starting Config: N=${N_SIZE}, Density=${DENSITY}"
        
        # Loop over iterations
        for i in $(seq 1 $TESTS_PER_CONFIG); do
            
            # Create a highly specific filename to avoid any chance of conflicts
            MATRIX_FILE="matrices/test_matrix_w${WORKERS}_N${N_SIZE}_d${DENSITY}_job${SLURM_JOB_ID}_iter${i}.mtx"
            
            # 1. Generate the matrix
            python3 matrices/gen_sbatch.py $N_SIZE $DENSITY $MATRIX_FILE

            # 2. Reconstruct Test
            START_REC=$(date +%s.%N)
            srun -n $WORKERS ./src/example_test.my $MATRIX_FILE 1
            REC_CODE=$?
            END_REC=$(date +%s.%N)
            
            REC_TIME=$(awk -v s=$START_REC -v e=$END_REC 'BEGIN {printf "%.4f", e-s}')
            echo "RUNTIME_RECONSTRUCT_${N_SIZE}_${DENSITY}=$REC_TIME"

            # 3. Vector Test
            START_VEC=$(date +%s.%N)
            srun -n $WORKERS ./src/example_test.my $MATRIX_FILE 2
            VEC_CODE=$?
            END_VEC=$(date +%s.%N)
            
            VEC_TIME=$(awk -v s=$START_VEC -v e=$END_VEC 'BEGIN {printf "%.4f", e-s}')
            echo "RUNTIME_VECTOR_${N_SIZE}_${DENSITY}=$VEC_TIME"

            # 4. Cleanup
            rm $MATRIX_FILE

            # 5. Check for failure
            if [ $REC_CODE -ne 0 ] || [ $VEC_CODE -ne 0 ]; then
                echo "TEST FAILED on N=$N_SIZE Density=$DENSITY iteration $i"
            fi

        done
    done
done

echo "ALL TESTS COMPLETED FOR $WORKERS WORKERS"
