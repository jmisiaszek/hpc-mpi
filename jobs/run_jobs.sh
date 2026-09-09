#!/bin/bash
#SBATCH --job-name=ilu_scale
#SBATCH --time=04:00:00        # see note below -- 1M-row matrices may need more than this
#SBATCH --output=logs/ilu_scale_j%j.out   # %j = job ID (unique regardless of ntasks)

# NOTE: dropped %n from the old --output pattern. In Slurm's filename
# patterns, %n means "node index within this job" (almost always 0 unless
# you span multiple nodes), NOT number of tasks -- so the old filenames were
# not actually distinguishing worker counts the way the comment assumed.
# We log the real worker count explicitly inside the script instead (see
# WORKERS= line below), which is what parse_results.py relies on.

source .venv/bin/activate   # keep if you still need the venv for anything else;
                            # not required by this script itself since matrices
                            # are pre-generated, not generated here.

WORKERS=$SLURM_NTASKS
TEST_BIN=./src/test
MATRICES=(1m2d.mtx 1m3d.mtx)
PERF_REPS=20     # repetitions *inside* the binary for solve/multiply averaging
REPEATS=3        # separate full re-runs per matrix, for median-based noise reduction

echo "=== Starting Scaling Test ==="
echo "WORKERS=$WORKERS"
echo "PERF_REPS=$PERF_REPS"
echo "REPEATS=$REPEATS"
echo "=============================="

for MATRIX_FILE in "${MATRICES[@]}"; do
    echo ">> RUNNING_MATRIX=$MATRIX_FILE"

    # Correctness check once per matrix per worker count.
    # "vector" mode is cheap even at N=1e6 -- do NOT use reconstruct
    # (mode "1" in the old binary) at this scale, it's O(N) comm rounds.
    srun -n $WORKERS $TEST_BIN ./matrices/$MATRIX_FILE vector --trials 1
    VEC_CODE=$?
    if [ $VEC_CODE -ne 0 ]; then
        echo "CORRECTNESS FAILED for $MATRIX_FILE at WORKERS=$WORKERS"
    fi

    # Performance runs, repeated for noise averaging (median taken during parsing).
    for i in $(seq 1 $REPEATS); do
        echo ">> PERF_REP=$i"
        srun -n $WORKERS $TEST_BIN ./matrices/$MATRIX_FILE perf --perf-reps $PERF_REPS
        PERF_CODE=$?
        if [ $PERF_CODE -ne 0 ]; then
            echo "PERF RUN FAILED for $MATRIX_FILE rep $i at WORKERS=$WORKERS"
        fi
    done
done

echo "ALL TESTS COMPLETED FOR $WORKERS WORKERS"