#!/usr/bin/env python3
"""Parse ILU scaling logs and tabulate timings by worker count.

Reads the .out files written by run_jobs.sh, groups PERF lines by matrix and
worker count, takes the median across repeats, and prints a table per matrix
with speedup and parallel efficiency relative to the smallest worker count.

Usage:
    ./parse_results.py [logdir] [--csv out.csv] [--baseline N] [--phase P]
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

WORKERS_RE = re.compile(r'^WORKERS=(\d+)')
MATRIX_RE = re.compile(r'^>> RUNNING_MATRIX=(\S+)')
PERF_RE = re.compile(r'^PERF,phase=(\w+),(.*)$')
FAIL_RE = re.compile(r'^(CORRECTNESS|PERF RUN) (FAILED|TIMED OUT) for (\S+)')
PASS_RE = re.compile(r'^(TEST PASSED|OVERALL: PASSED)')

PHASE_ORDER = ['factorize', 'multiply', 'solve']


def parse_file(path):
    """Return (workers, samples, problems, meta) for one log file.

    samples:  {(matrix, phase): [times...]}
    problems: list of human-readable warnings
    meta:     {matrix: {'N': int, 'nnz': int}}
    """
    workers = None
    matrix = None
    samples = defaultdict(list)
    problems = []
    meta = {}
    saw_pass = False

    with open(path, errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')

            m = WORKERS_RE.match(line)
            if m:
                workers = int(m.group(1))
                continue

            m = MATRIX_RE.match(line)
            if m:
                matrix = m.group(1)
                continue

            m = FAIL_RE.match(line)
            if m:
                problems.append(f'{m.group(1).lower()} {m.group(2).lower()}: {m.group(3)}')
                continue

            if PASS_RE.match(line):
                saw_pass = True
                continue

            m = PERF_RE.match(line)
            if not m:
                continue

            phase = m.group(1)
            fields = {}
            for kv in m.group(2).split(','):
                if '=' not in kv:
                    continue
                k, v = kv.split('=', 1)
                fields[k] = v

            if 'time' not in fields:
                continue
            key_matrix = matrix or '<unknown>'
            samples[(key_matrix, phase)].append(float(fields['time']))

            info = meta.setdefault(key_matrix, {})
            for k in ('N', 'nnz'):
                if k in fields:
                    info[k] = int(fields[k])

    if workers is None:
        problems.append('no WORKERS= line found')
    if not samples and not problems:
        problems.append('no PERF lines found')
    if not saw_pass and samples:
        problems.append('no PASSED line seen')

    return workers, samples, problems, meta


def collect(logdir):
    """Merge every log file into {matrix: {phase: {workers: median_time}}}."""
    paths = sorted(glob.glob(os.path.join(logdir, '*.out')))
    if not paths:
        sys.exit(f'No .out files in {logdir}')

    times = defaultdict(lambda: defaultdict(dict))
    counts = defaultdict(lambda: defaultdict(dict))
    meta = {}
    issues = []

    for path in paths:
        workers, samples, problems, file_meta = parse_file(path)
        name = os.path.basename(path)

        for p in problems:
            issues.append(f'{name}: {p}')
        if workers is None:
            continue

        for matrix, info in file_meta.items():
            meta.setdefault(matrix, {}).update(info)

        for (matrix, phase), vals in samples.items():
            if workers in times[matrix][phase]:
                issues.append(
                    f'{name}: duplicate result for {matrix}/{phase} at P={workers}, '
                    f'keeping the faster one')
                prev = times[matrix][phase][workers]
                times[matrix][phase][workers] = min(prev, statistics.median(vals))
            else:
                times[matrix][phase][workers] = statistics.median(vals)
            counts[matrix][phase][workers] = len(vals)

    return times, counts, meta, issues


def fmt_table(rows, headers, aligns=None):
    cols = len(headers)
    aligns = aligns or ['>'] * cols
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        return '  '.join(f'{c:{aligns[i]}{widths[i]}}' for i, c in enumerate(cells))

    out = [line(headers), '  '.join('-' * w for w in widths)]
    out.extend(line(r) for r in rows)
    return '\n'.join(out)


def report(times, counts, meta, baseline_arg, phases):
    for matrix in sorted(times):
        phase_map = times[matrix]
        present = [p for p in phases if p in phase_map]
        if not present:
            continue

        workers = sorted({w for p in present for w in phase_map[p]})
        if not workers:
            continue

        baseline = baseline_arg if baseline_arg in workers else workers[0]

        info = meta.get(matrix, {})
        desc = ', '.join(f'{k}={v}' for k, v in sorted(info.items()))
        title = f'{matrix}' + (f'  ({desc})' if desc else '')
        print()
        print(title)
        print('=' * len(title))

        headers = ['P']
        aligns = ['>']
        for p in present:
            headers += [f'{p} (s)', 'speedup', 'eff']
            aligns += ['>', '>', '>']

        rows = []
        for w in workers:
            row = [str(w)]
            for p in present:
                t = phase_map[p].get(w)
                base = phase_map[p].get(baseline)
                if t is None:
                    row += ['-', '-', '-']
                    continue
                n = counts[matrix][p].get(w, 0)
                mark = '' if n >= 3 else f'~{n}'
                row.append(f'{t:.4f}{mark}')
                if base:
                    sp = base / t
                    eff = sp / (w / baseline)
                    row += [f'{sp:.2f}x', f'{eff * 100:.0f}%']
                else:
                    row += ['-', '-']
            rows.append(row)

        print(fmt_table(rows, headers, aligns))
        print(f'median of repeats; speedup and efficiency relative to P={baseline}')
        print('a ~N suffix means only N repeat(s) were available')


def write_csv(path, times, counts, meta):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['matrix', 'N', 'nnz', 'phase', 'workers', 'median_time_s', 'n_repeats'])
        for matrix in sorted(times):
            info = meta.get(matrix, {})
            for phase in sorted(times[matrix]):
                for workers in sorted(times[matrix][phase]):
                    w.writerow([
                        matrix,
                        info.get('N', ''),
                        info.get('nnz', ''),
                        phase,
                        workers,
                        f'{times[matrix][phase][workers]:.6f}',
                        counts[matrix][phase].get(workers, 0),
                    ])
    print(f'\nwrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logdir', nargs='?', default='logs')
    ap.add_argument('--csv', metavar='FILE', help='also write raw numbers to CSV')
    ap.add_argument('--baseline', type=int, default=1,
                    help='worker count for speedup baseline (default 1, '
                         'falls back to the smallest available)')
    ap.add_argument('--phase', action='append', dest='phases',
                    help='restrict to a phase; repeatable')
    args = ap.parse_args()

    phases = args.phases or PHASE_ORDER

    times, counts, meta, issues = collect(args.logdir)
    report(times, counts, meta, args.baseline, phases)

    if issues:
        print('\nWarnings')
        print('========')
        for i in issues:
            print(f'  {i}')

    if args.csv:
        write_csv(args.csv, times, counts, meta)


if __name__ == '__main__':
    main()