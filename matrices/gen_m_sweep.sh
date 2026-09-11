#!/bin/bash
# Generate a family of k=7 banded matrices that differ ONLY in the largest
# offset. N and nnz are held constant so any timing difference is attributable
# to the band width, which is what sets the separator size in DAS-ILU.
#
# Disk: each matrix is ~7M lines, roughly 140MB. The full sweep is ~1.2GB.
set -e

N=${N:-1000000}
OUTDIR=${OUTDIR:-matrices}
OFFSETS=${OFFSETS:-"4 16 64 256 1024 4096 16384 65536 131072"}

mkdir -p "$OUTDIR"

for m in $OFFSETS; do
    out="$OUTDIR/band_n${N}_m${m}.mtx"
    if [ -f "$out" ]; then
        echo "skip (exists): $out"
        continue
    fi
    python3 matrices/gen_banded.py --n "$N" --offset "$m" -o "$out"
done

echo
echo "Block size at P=16 is $((N / 16)) rows."
echo "Offsets above that make a rank depend on more than one lower rank."