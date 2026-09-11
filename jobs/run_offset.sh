#!/bin/bash
#SBATCH --job-name=ilu_offset
#SBATCH --time=04:00:00
#SBATCH --output=logs/ilu_offset_j%j.out

source .venv/bin/activate

WORKERS=$SLURM_NTASKS
TEST_BIN=./src/test.my
N=${N:-1000000}
OFFSETS=${OFFSETS:-"4 16 64 256 1024 4096 16384 65536 131072"}
PERF_REPS=20
REPEATS=3

SRUN_FLAGS="--unbuffered --kill-on-bad-exit=1"
RUN_TIMEOUT="timeout -k 30 900"

echo "=== Starting Scaling Test ==="
echo "WORKERS=$WORKERS"
echo "PERF_REPS=$PERF_REPS"
echo "REPEATS=$REPEATS"
echo "BINARY=$TEST_BIN"
ls -l --time-style=full-iso "$TEST_BIN"
echo "NODES=$SLURM_JOB_NUM_NODES"
echo "TASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "BLOCK_ROWS=$((N / WORKERS))"
echo "=============================="

for m in $OFFSETS; do
    MATRIX_FILE="band_n${N}_m${m}.mtx"
    if [ ! -f "./matrices/$MATRIX_FILE" ]; then
        echo "SKIP missing matrix $MATRIX_FILE"
        continue
    fi

    echo ">> RUNNING_MATRIX=$MATRIX_FILE"

    $RUN_TIMEOUT srun -n $WORKERS $SRUN_FLAGS \
        $TEST_BIN ./matrices/$MATRIX_FILE vector --trials 1
    VEC_CODE=$?
    if [ $VEC_CODE -eq 124 ]; then
        echo "CORRECTNESS TIMED OUT for $MATRIX_FILE at WORKERS=$WORKERS"
        continue
    elif [ $VEC_CODE -ne 0 ]; then
        echo "CORRECTNESS FAILED for $MATRIX_FILE at WORKERS=$WORKERS (code $VEC_CODE)"
        continue
    fi

    for i in $(seq 1 $REPEATS); do
        echo ">> PERF_REP=$i"
        $RUN_TIMEOUT srun -n $WORKERS $SRUN_FLAGS \
            $TEST_BIN ./matrices/$MATRIX_FILE perf --perf-reps $PERF_REPS
        PERF_CODE=$?
        if [ $PERF_CODE -eq 124 ]; then
            echo "PERF RUN TIMED OUT for $MATRIX_FILE rep $i at WORKERS=$WORKERS"
        elif [ $PERF_CODE -ne 0 ]; then
            echo "PERF RUN FAILED for $MATRIX_FILE rep $i at WORKERS=$WORKERS"
        fi
    done
done

echo "ALL TESTS COMPLETED FOR $WORKERS WORKERS"