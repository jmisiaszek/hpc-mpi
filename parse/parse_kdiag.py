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
NODES_RE = re.compile(r'^NODES=(\d+)')
MATRIX_RE = re.compile(r'^>> RUNNING_MATRIX=(\S+)')
PERF_RE = re.compile(r'^PERF,phase=(\w+),(.*)$')
FAIL_RE = re.compile(r'^(CORRECTNESS|PERF RUN) (FAILED|TIMED OUT) for (\S+)')
PASS_RE = re.compile(r'^(TEST PASSED|OVERALL: PASSED)')

PHASE_ORDER = ['factorize', 'multiply', 'solve']
OFFSET_RE = re.compile(r'_m(\d+)\.mtx$')


def sort_key(matrix):
    """Sort band_*_m<K>.mtx numerically by offset; everything else by name."""
    m = OFFSET_RE.search(matrix)
    return (0, int(m.group(1)), '') if m else (1, 0, matrix)


def offset_label(matrix):
    m = OFFSET_RE.search(matrix)
    return m.group(1) if m else matrix


def parse_file(path):
    """Return (workers, samples, problems, meta) for one log file.

    samples:  {(matrix, phase): [times...]}
    problems: list of human-readable warnings
    meta:     {matrix: {'N': int, 'nnz': int}}
    """
    workers = None
    nodes = None
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

            m = NODES_RE.match(line)
            if m:
                nodes = int(m.group(1))
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

    return workers, nodes, samples, problems, meta


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
        workers, nodes, samples, problems, file_meta = parse_file(path)
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
                    f'{name}: duplicate result for {matrix}/{phase} at P={workers} '
                    f'(NODES={nodes}) -- runs with different node counts collide here; '
                    f'parse them from separate directories. Keeping the faster one')
                prev = times[matrix][phase][workers]
                times[matrix][phase][workers] = min(prev, statistics.median(vals))
            else:
                times[matrix][phase][workers] = statistics.median(vals)
            counts[matrix][phase][workers] = len(vals)

    return times, counts, meta, issues


def fmt_table_md(rows, headers, aligns=None):
    """GitHub-flavoured Markdown table; every column right-aligned but the first."""
    sep = ['---:' if i else '---' for i in range(len(headers))]
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(sep) + '|']
    out.extend('| ' + ' | '.join(r) + ' |' for r in rows)
    return '\n'.join(out)


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


def report_by_matrix(times, counts, meta, phases, markdown=False):
    """One table per worker count, rows = matrix (offset), columns = phases.

    This is the view for an offset sweep: P is held fixed and the band width
    varies, so the rows show how the largest offset drives each phase.
    """
    matrices = sorted(times, key=sort_key)
    banded = any(OFFSET_RE.search(m) for m in matrices)
    first_col = 'max offset' if banded else 'matrix'

    workers = sorted({w for m in matrices for p in times[m] for w in times[m][p]})

    for wcount in workers:
        present = [p for p in phases
                   if any(wcount in times[m].get(p, {}) for m in matrices)]
        if not present:
            continue

        title = f'P = {wcount}'
        print()
        if markdown:
            print(f'### {title}')
            print()
        else:
            print(title)
            print('=' * len(title))

        headers = [first_col] + [f'{p} (s)' for p in present]
        aligns = ['>'] * len(headers)

        rows = []
        for matrix in matrices:
            cells = [offset_label(matrix)]
            any_val = False
            for p in present:
                t = times[matrix].get(p, {}).get(wcount)
                if t is None:
                    cells.append('-')
                    continue
                any_val = True
                n = counts[matrix][p].get(wcount, 0)
                mark = '' if n >= 3 else (f' ~{n}' if markdown else f'~{n}')
                cells.append(f'{t:.4f}{mark}')
            if any_val:
                rows.append(cells)

        if not rows:
            continue

        if markdown:
            print(fmt_table_md(rows, headers))
            print()
            print('Medians across repeats. `~N` marks a cell with only N repeat(s).')
        else:
            print(fmt_table(rows, headers, aligns))
            print('median of repeats')


def report(times, counts, meta, baseline_arg, phases, markdown=False):
    for matrix in sorted(times, key=sort_key):
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
        if markdown:
            print(f'### {title}')
            print()
        else:
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
                mark = '' if n >= 3 else (f' ~{n}' if markdown else f'~{n}')
                row.append(f'{t:.4f}{mark}')
                if base:
                    sp = base / t
                    eff = sp / (w / baseline)
                    row += [f'{sp:.2f}x', f'{eff * 100:.0f}%']
                else:
                    row += ['-', '-']
            rows.append(row)

        if markdown:
            print(fmt_table_md(rows, headers))
            print()
            print(f'Medians across repeats; speedup and efficiency relative to P={baseline}. '
                  '`~N` marks a cell with only N repeat(s).')
        else:
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
    ap.add_argument('--md', action='store_true',
                    help='emit Markdown tables instead of plain text')
    ap.add_argument('--by-matrix', action='store_true',
                    help='one table per worker count with matrices as rows; '
                         'use this for an offset sweep')
    args = ap.parse_args()

    phases = args.phases or PHASE_ORDER

    times, counts, meta, issues = collect(args.logdir)
    if args.by_matrix:
        report_by_matrix(times, counts, meta, phases, markdown=args.md)
    else:
        report(times, counts, meta, args.baseline, phases, markdown=args.md)

    if issues:
        if args.md:
            print('\n### Warnings\n')
            for i in issues:
                print(f'- {i}')
        else:
            print('\nWarnings')
            print('========')
            for i in issues:
                print(f'  {i}')

    if args.csv:
        write_csv(args.csv, times, counts, meta)


if __name__ == '__main__':
    main()