#!/usr/bin/env python3
"""Generate k-diagonal (banded) matrices in MatrixMarket coordinate format.

The point of this generator is to vary the *largest offset* while holding the
number of diagonals k, the dimension N, and the nonzero count essentially
constant. In DAS-ILU the largest offset is what determines how many rows of
each rank's block reach below its first row -- i.e. the separator size -- so
it is the parameter that drives message sizes, the convergence exchange and
the serial solve chain.

Default offset pattern is {0, +/-1, +/-2, +/-m}: k=7 diagonals with a single
tunable m, so a sweep over m changes only the band width.

The matrix is made strictly diagonally dominant (diagonal = sum of off-diagonal
magnitudes + margin) because this factorization does no pivoting; without
dominance you risk tiny pivots and a convergence loop that never terminates.

Usage:
    ./gen_banded.py --n 1000000 --offset 4096 -o matrices/band_m4096.mtx
    ./gen_banded.py --n 1000000 --offset 4096 --near 1 2 -o out.mtx
"""

import argparse
import os
import sys


def build_offsets(near, m):
    """Return the sorted, deduplicated nonzero offsets (excluding the diagonal)."""
    offs = set()
    for d in near:
        if d == 0:
            continue
        offs.add(d)
        offs.add(-d)
    if m != 0:
        offs.add(m)
        offs.add(-m)
    return sorted(offs)


def count_nnz(n, offsets):
    total = n  # diagonal
    for d in offsets:
        total += n - abs(d) if abs(d) < n else 0
    return total


def generate(path, n, offsets, offdiag, margin, chunk_rows=20000):
    nnz = count_nnz(n, offsets)
    diag = abs(offdiag) * len(offsets) + margin

    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write('%%MatrixMarket matrix coordinate real general\n')
        fh.write(f'% banded, k={len(offsets) + 1}, offsets={[0] + offsets}\n')
        fh.write(f'% diagonally dominant: diag={diag}, offdiag={offdiag}\n')
        fh.write(f'{n} {n} {nnz}\n')

        buf = []
        written = 0
        for i in range(n):
            # 1-indexed output
            r = i + 1
            buf.append(f'{r} {r} {diag}\n')
            written += 1
            for d in offsets:
                j = i + d
                if 0 <= j < n:
                    buf.append(f'{r} {j + 1} {offdiag}\n')
                    written += 1
            if i % chunk_rows == 0 and buf:
                fh.write(''.join(buf))
                buf = []
        if buf:
            fh.write(''.join(buf))

    if written != nnz:
        os.unlink(tmp)
        sys.exit(f'internal error: wrote {written} entries, header said {nnz}')

    os.replace(tmp, path)
    return nnz, diag


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, required=True, help='matrix dimension')
    ap.add_argument('--offset', type=int, required=True,
                    help='largest offset m (the band half-width)')
    ap.add_argument('--near', type=int, nargs='*', default=[1, 2],
                    help='near-diagonal offsets, mirrored (default: 1 2)')
    ap.add_argument('--offdiag', type=float, default=-1.0,
                    help='value on every off-diagonal (default -1)')
    ap.add_argument('--margin', type=float, default=0.1,
                    help='diagonal dominance margin (default 0.1)')
    ap.add_argument('-o', '--out', required=True, help='output .mtx path')
    args = ap.parse_args()

    offsets = build_offsets(args.near, args.offset)
    if not offsets:
        sys.exit('no off-diagonals requested')
    if args.offset >= args.n:
        sys.exit(f'offset {args.offset} must be smaller than n={args.n}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    nnz, diag = generate(args.out, args.n, offsets, args.offdiag, args.margin)

    print(f'{args.out}: N={args.n} k={len(offsets) + 1} max_offset={args.offset} '
          f'nnz={nnz} diag={diag}')


if __name__ == '__main__':
    main()