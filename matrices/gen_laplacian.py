#!/usr/bin/env python3
"""
Generate a discrete Laplacian (finite-difference stencil) matrix on a
1D/2D/3D grid, written in MatrixMarket coordinate format compatible with
read_matrix() in example_test.cpp (it just skips '%' comment lines and
reads "N M nnz" then "row col val" triplets, 1-indexed).

Grid points are linearized in row-major (C) order: for a grid of shape
(d0, d1, ..., dk-1), point (i0, i1, ..., ik-1) maps to global row/col index
    idx = i0*stride0 + i1*stride1 + ... + i(k-1)
where the LAST axis varies fastest (standard C ordering). This ordering
matters a lot for how well the matrix lines up with contiguous row-block
partitioning -- see the notes below the script.

Usage examples:
    # 1D chain, N=1000
    python3 gen_laplacian.py --shape 1000 -o lap1d_1000.mtx

    # 2D grid, 100x100 = 10000 unknowns, standard 5-point stencil
    python3 gen_laplacian.py --shape 100,100 -o lap2d_100x100.mtx

    # 3D grid, 30x30x30 = 27000 unknowns, 7-point stencil
    python3 gen_laplacian.py --shape 30,30,30 -o lap3d_30.mtx

    # 2D, periodic boundary conditions (wraps around)
    python3 gen_laplacian.py --shape 64,64 --periodic -o lap2d_periodic.mtx

    # 2D, weaken diagonal dominance to stress convergence (more ILU iters)
    python3 gen_laplacian.py --shape 100,100 --diag-shift 0.01 -o lap2d_weak.mtx

    # 2D, transpose the ordering (swap fast/slow axis) to create a
    # partitioning-hostile layout for contiguous row-block partitioning
    python3 gen_laplacian.py --shape 100,100 --axis-order 1,0 -o lap2d_bad.mtx
"""

import argparse
import numpy as np
import sys


def build_laplacian(shape, periodic=False, diag_shift=0.0, axis_order=None,
                     anisotropy=None):
    """
    shape: tuple of grid dimensions, e.g. (nx,), (ny, nx), (nz, ny, nx)
    periodic: bool, wrap around each axis instead of a Dirichlet boundary
    diag_shift: added uniformly to every diagonal entry, controls diagonal
                dominance independent of grid connectivity
    axis_order: permutation of range(len(shape)) controlling which axis
                varies fastest in the linear index (default: last axis
                fastest, i.e. identity order for standard row-major)
    anisotropy: per-axis multiplier on that axis's off-diagonal weight
                (default: all 1.0)

    Returns (N, rows, cols, vals) as numpy arrays, 0-indexed.
    """
    dims = list(shape)
    ndim = len(dims)
    N = int(np.prod(dims))

    if axis_order is None:
        axis_order = list(range(ndim))
    if anisotropy is None:
        anisotropy = [1.0] * ndim

    # Strides for linear indexing given the chosen fast/slow axis order.
    # axis_order[-1] is the fastest-varying axis.
    strides = [0] * ndim
    running = 1
    for ax in reversed(axis_order):
        strides[ax] = running
        running *= dims[ax]

    # Coordinate grids for every point, one array per axis.
    coords = np.indices(dims)  # shape: (ndim, *dims)
    coords = coords.reshape(ndim, -1)  # (ndim, N)

    lin_idx = np.zeros(N, dtype=np.int64)
    for ax in range(ndim):
        lin_idx += coords[ax] * strides[ax]

    rows_list = [lin_idx]
    cols_list = [lin_idx]
    vals_list = [np.zeros(N)]  # placeholder, filled in after we know degree

    degree = np.zeros(N)

    for ax in range(ndim):
        w = anisotropy[ax]
        for delta in (-1, 1):
            new_coord = coords[ax] + delta
            if periodic:
                new_coord = np.mod(new_coord, dims[ax])
                valid = np.ones(N, dtype=bool)
            else:
                valid = (new_coord >= 0) & (new_coord < dims[ax])

            neigh_coords = coords.copy()
            neigh_coords[ax] = np.where(valid, new_coord, 0)  # dummy for invalid

            neigh_lin = np.zeros(N, dtype=np.int64)
            for a2 in range(ndim):
                neigh_lin += neigh_coords[a2] * strides[a2]

            r = lin_idx[valid]
            c = neigh_lin[valid]
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(np.full(r.shape, -w))

            degree[valid] += w

    vals_list[0] = degree + diag_shift  # diagonal entries

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    return N, rows, cols, vals


def write_matrix_market(path, N, rows, cols, vals):
    nnz = len(rows)
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {nnz}\n")
        # 1-indexed, as read_matrix() expects (it decrements on read).
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r + 1} {c + 1} {v:.10g}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", required=True,
                    help="Comma-separated grid dims, e.g. '1000' (1D), "
                         "'100,100' (2D), '30,30,30' (3D)")
    p.add_argument("-o", "--output", required=True, help="Output .mtx path")
    p.add_argument("--periodic", action="store_true",
                    help="Periodic (wraparound) boundary conditions")
    p.add_argument("--diag-shift", type=float, default=0.0,
                    help="Extra value added to every diagonal entry "
                         "(controls diagonal dominance / conditioning)")
    p.add_argument("--axis-order", default=None,
                    help="Comma-separated permutation of axis indices, "
                         "last one varies fastest. Default: natural order "
                         "(last axis of --shape varies fastest). Use this "
                         "to test partition-hostile orderings.")
    p.add_argument("--anisotropy", default=None,
                    help="Comma-separated per-axis off-diagonal weight, "
                         "e.g. '1,1' for isotropic 2D or '1,5' to make "
                         "one direction much stiffer than the other")
    args = p.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    axis_order = ([int(x) for x in args.axis_order.split(",")]
                  if args.axis_order else None)
    anisotropy = ([float(x) for x in args.anisotropy.split(",")]
                  if args.anisotropy else None)

    N, rows, cols, vals = build_laplacian(
        shape, periodic=args.periodic, diag_shift=args.diag_shift,
        axis_order=axis_order, anisotropy=anisotropy)

    write_matrix_market(args.output, N, rows, cols, vals)

    print(f"Wrote {args.output}: N={N}, nnz={len(rows)}, "
          f"avg nnz/row={len(rows)/N:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()