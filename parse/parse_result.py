#!/usr/bin/env python3
"""
Parse Slurm .out logs written by run_scaling.sh into a summary CSV with
median time per (matrix, phase, rank count).

Doesn't rely on filenames for the rank count -- the binary already embeds
P=... in every PERF line. Instead we track which matrix is "current" by
watching for the RUNNING_MATRIX= marker the sbatch script prints before
each srun call.

Usage:
    python3 parse_results.py [--logdir logs] [--out results.csv]
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

PERF_RE = re.compile(
    r"PERF,phase=(\w+),N=(\d+)(?:,nnz=(\d+))?,P=(\d+),time=([\d.eE+-]+)"
)
MATRIX_RE = re.compile(r"RUNNING_MATRIX=(\S+)")
FAIL_RE = re.compile(r"(CORRECTNESS FAILED|PERF RUN FAILED).*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    pattern = os.path.join(args.logdir, "ilu_scale_j*.out")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No log files found matching {pattern}", file=sys.stderr)
        sys.exit(1)

    data = defaultdict(list)          # (matrix, phase, P) -> [times]
    meta = {}                         # (matrix, phase, P) -> (N, nnz)
    failures = []

    for path in files:
        current_matrix = None
        with open(path) as f:
            for line in f:
                mm = MATRIX_RE.search(line)
                if mm:
                    current_matrix = mm.group(1)
                    continue

                fm = FAIL_RE.search(line)
                if fm:
                    failures.append(f"{os.path.basename(path)}: {line.strip()}")
                    continue

                pm = PERF_RE.search(line)
                if pm and current_matrix is not None:
                    phase, N, nnz, P, t = pm.groups()
                    key = (current_matrix, phase, int(P))
                    data[key].append(float(t))
                    meta[key] = (N, nnz)

    if failures:
        print("=== FAILURES DETECTED ===", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        print(file=sys.stderr)

    rows = []
    for (matname, phase, P), times in sorted(data.items()):
        N, nnz = meta[(matname, phase, P)]
        med = statistics.median(times)
        rows.append({
            "matrix": matname, "phase": phase, "ranks": P,
            "N": N, "nnz": nnz,
            "median_time": f"{med:.6f}",
            "n_samples": len(times),
            "all_times": ";".join(f"{t:.6f}" for t in times),
        })
        print(f"{matname:10s} phase={phase:10s} P={P:3d}  median={med:.6f}s  "
              f"(n={len(times)})")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "matrix", "phase", "ranks", "N", "nnz",
            "median_time", "n_samples", "all_times"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {args.out}")
    if failures:
        print(f"({len(failures)} failure(s) logged above -- treat those "
              f"configs' numbers with suspicion)", file=sys.stderr)


if __name__ == "__main__":
    main()