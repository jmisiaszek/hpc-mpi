mkdir -p logs
for w in {1..10}; do
    sbatch --ntasks=$w jobs/run_tests_loop.sh
done