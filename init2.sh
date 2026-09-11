mkdir -p logs_1node logs_2node

for P in 16 32 48; do
    sbatch --ntasks=$P --nodes=1 \
           --output=logs_1node/place_P${P}_j%j.out \
           jobs/run_offset_sweep.sh

    sbatch --ntasks=$P --nodes=2 --ntasks-per-node=$((P/2)) \
           --output=logs_2node/place_P${P}_j%j.out \
           jobs/run_offset_sweep.sh
done