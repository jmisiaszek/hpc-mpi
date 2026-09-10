#!/bin/bash
#SBATCH --job-name=ilu_scale
#SBATCH --time=04:00:00
#SBATCH --output=logs/ilu_scale_j%j.out

source .venv/bin/activate

WORKERS=$SLURM_NTASKS
TEST_BIN=./src/test.my
MATRICES=(1m2d.mtx 1m3d.mtx)
PERF_REPS=20
REPEATS=3

# --unbuffered: forward task stdout/stderr line-by-line instead of block
#   buffering it. Without this a hung run shows nothing at all.
# --kill-on-bad-exit: don't leave 15 ranks spinning when one aborts.
SRUN_FLAGS="--unbuffered --kill-on-bad-exit=1"

# Wall-clock cap per invocation, so a hang fails fast instead of eating the
# whole 4h allocation. timeout sends TERM; -k 30 follows with KILL.
RUN_TIMEOUT="timeout -k 30 600"

echo "=== Starting Scaling Test ==="
echo "WORKERS=$WORKERS"
echo "PERF_REPS=$PERF_REPS"
echo "REPEATS=$REPEATS"
echo "BINARY=$TEST_BIN"
ls -l --time-style=full-iso "$TEST_BIN"   # confirm you are running a fresh build
echo "=============================="

for MATRIX_FILE in "${MATRICES[@]}"; do
    echo ">> RUNNING_MATRIX=$MATRIX_FILE"

    $RUN_TIMEOUT srun -n $WORKERS $SRUN_FLAGS \
        $TEST_BIN ./matrices/$MATRIX_FILE vector --trials 1
    VEC_CODE=$?
    if [ $VEC_CODE -eq 124 ]; then
        echo "CORRECTNESS TIMED OUT for $MATRIX_FILE at WORKERS=$WORKERS"
        continue
    elif [ $VEC_CODE -ne 0 ]; then
        echo "CORRECTNESS FAILED for $MATRIX_FILE at WORKERS=$WORKERS (code $VEC_CODE)"
        continue    # no point timing a broken factorization
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