// Benchmark driver for the ILU assignment (new solution only, no
// correctness probing - see ../tests/test_full.cpp for that).
//
// Times ILU_factorize once, then ILU_solve and ILU_multiply averaged over
// many calls (the assignment says to expect many solves per factorization).
// A cheap sanity check multiply(solve(v)) ~= v guards against benchmarking
// a broken build. Rank 0 prints a single machine-parseable line:
//
//   BENCH N=... nnz=... np=... factorize_s=... solve_s=... multiply_s=... sanity_err=...
//
// Usage: bench <matrix.mtx> [n_solve]   (default n_solve = 50)

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mpi.h>

#include "ilu.h"

static int first_row(int rank, int N, int world_size)
{
    return (int)((long long)rank * N / world_size);
}

// Reads matrix in a MatrixMarket format (same reader as example_test.cpp)
static void read_matrix(const char* in_file, int* N, int* nnz,
                        int** row, int** col, double** val)
{
    FILE* fp = fopen(in_file, "r");
    char* line = NULL;
    size_t len = 0;
    int l = -1;
    if (fp == NULL) { fprintf(stderr, "cannot open %s\n", in_file); exit(1); }

    while (getline(&line, &len, fp) != -1) {
        if (line[0] == '%') continue;
        if (l == -1) {
            int M;
            sscanf(line, "%d %d %d", N, &M, nnz);
            assert(M == *N);
            *row = (int*)malloc(*nnz * sizeof(int));
            *col = (int*)malloc(*nnz * sizeof(int));
            *val = (double*)malloc(*nnz * sizeof(double));
        } else {
            sscanf(line, "%d %d %lf", *row + l, *col + l, *val + l);
            (*row)[l]--;
            (*col)[l]--;
        }
        l++;
    }
    fclose(fp);
    if (line) free(line);
}

int main(int argc, char* argv[])
{
    assert(argc >= 2);
    int n_solve = (argc >= 3) ? atoi(argv[2]) : 50;
    assert(n_solve > 0);

    MPI_Init(&argc, &argv);
    int rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    int N = 0, nnz = 0;
    int *row = NULL, *col = NULL;
    double* val = NULL;
    if (rank == 0)
        read_matrix(argv[1], &N, &nnz, &row, &col, &val);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    struct ILUFact* ilu = ILU_factorize(N, nnz, row, col, val);
    double t_fact = MPI_Wtime() - t0;

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int lo = first_row(rank, N, ws);
    int nl = first_row(rank + 1, N, ws) - lo;

    std::vector<double> b(nl), x(nl), back(nl);
    for (int i = 0; i < nl; ++i)
        b[i] = 1.0 + 0.5 * std::sin(0.01 * (lo + i));

    // Sanity: multiply(solve(v)) ~= v, so the timings below are for a
    // working factorization, not garbage. Also serves as a warmup.
    ILU_solve(ilu, b.data(), x.data());
    ILU_multiply(ilu, x.data(), back.data());
    double loc = 0.0;
    for (int i = 0; i < nl; ++i) {
        double d = std::fabs(back[i] - b[i]);
        if (!std::isfinite(d)) d = 1e30;
        loc = std::max(loc, d);
    }
    double sanity_err = 0.0;
    MPI_Reduce(&loc, &sanity_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts = MPI_Wtime();
    for (int it = 0; it < n_solve; ++it)
        ILU_solve(ilu, b.data(), x.data());
    double t_solve = (MPI_Wtime() - ts) / n_solve;

    MPI_Barrier(MPI_COMM_WORLD);
    double tm = MPI_Wtime();
    for (int it = 0; it < n_solve; ++it)
        ILU_multiply(ilu, b.data(), x.data());
    double t_mult = (MPI_Wtime() - tm) / n_solve;

    double tf_max = 0.0, ts_max = 0.0, tm_max = 0.0;
    MPI_Reduce(&t_fact,  &tf_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_solve, &ts_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_mult,  &tm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("BENCH N=%d nnz=%d np=%d factorize_s=%.6f solve_s=%.9f "
               "multiply_s=%.9f sanity_err=%.3e\n",
               N, nnz, ws, tf_max, ts_max, tm_max, sanity_err);
        if (sanity_err >= 1e-4)
            fprintf(stderr, "WARNING: sanity check failed (err %.3e)\n", sanity_err);
    }

    ILU_free(ilu);
    free(row); free(col); free(val);
    MPI_Finalize();
    return 0;
}