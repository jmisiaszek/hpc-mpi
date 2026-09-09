mkdir -p logs
for w in 1 $(seq 4 4 48); do
    sbatch --ntasks=$w jobs/run_jobs.sh
done