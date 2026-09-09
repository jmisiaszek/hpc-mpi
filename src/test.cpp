#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <mpi.h>
#include <iostream>
#include <iomanip>

#include "ilu.h"

using namespace std;

#define EPS 10e-4

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int test_rank_first_row(int rank, int N, int world_size)
{
    return rank * N / world_size;
}

// Reads matrix in MatrixMarket format (comment lines starting with '%' are
// skipped; first data line is "N M nnz"; following lines are "row col val",
// 1-indexed).
void read_matrix(char* in_file, int* N, int* nnz, int** row, int** col, double** val)
{
    FILE *fp = fopen(in_file, "r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int l = -1;
    if (fp == NULL)
    {
        fprintf(stderr, "Could not open matrix file: %s\n", in_file);
        exit(EXIT_FAILURE);
    }

    while ((read = getline(&line, &len, fp)) != -1)
    {
        if (line[0] == '%') continue;
        if (l == -1)
        {
            int M;
            sscanf(line, "%d %d %d", N, &M, nnz);
            assert(M == *N);
            *row = (int*) malloc(*nnz * sizeof(int));
            *col = (int*) malloc(*nnz * sizeof(int));
            *val = (double*) malloc(*nnz * sizeof(double));
        }
        else
        {
            sscanf(line, "%d %d %lf", *row + l, *col + l, *val + l);
            // Fix numbering from 1
            (*row)[l]--;
            (*col)[l]--;
        }
        l++;
    }

    fclose(fp);
    if (line)
        free(line);
}

// A tiny helper for rank-0-only status printing.
void log0(int rank, const char* fmt, ...)
{
    if (rank != 0) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Correctness: reconstruct A from ILU_multiply column-by-column and compare
// against the original entries. O(N) communication rounds -- only use this
// on small/medium matrices (roughly N <= a few thousand). Do NOT run this on
// million-row matrices; it will not finish in reasonable time.
// ---------------------------------------------------------------------------
bool test_reconstruct(int N, int nnz, int* row, int* col, double* val, struct ILUFact* ilu)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int first_row = test_rank_first_row(rank, N, world_size);
    int last_row = test_rank_first_row(rank + 1, N, world_size);
    int n_local_rows = last_row - first_row;

    bool success = true;
    double* v = (double*) malloc(n_local_rows * sizeof(double));
    double* res = (double*) malloc(n_local_rows * sizeof(double));

    for (int t = 0; t < N; t++) {
        for (int i = 0; i < n_local_rows; i++) {
            v[i] = 0;
            res[i] = 0;
        }
        if (t >= first_row && t < last_row) {
            v[t - first_row] = 1;
        }

        ILU_multiply(ilu, v, res);

        int* recvcounts = NULL;
        int* displs = NULL;
        double* global_res = NULL;

        if (rank == 0) {
            recvcounts = (int*) malloc(world_size * sizeof(int));
            displs = (int*) malloc(world_size * sizeof(int));
            global_res = (double*) malloc(N * sizeof(double));

            for (int i = 0; i < world_size; i++) {
                int r_first = test_rank_first_row(i, N, world_size);
                int r_last = test_rank_first_row(i + 1, N, world_size);
                recvcounts[i] = r_last - r_first;
                displs[i] = r_first;
            }
        }

        MPI_Gatherv(res, n_local_rows, MPI_DOUBLE,
                    global_res, recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            double* original = (double*) malloc(N * sizeof(double));
            for (int i = 0; i < N; i++) {
                original[i] = 0;
            }
            for (int i = 0; i < nnz; i++) {
                if (col[i] == t) {
                    original[row[i]] = val[i];
                }
            }

            for (int i = 0; i < N; i++) {
                if (original[i] == 0.0) {
                    continue;
                }
                if (abs(original[i] - global_res[i]) > 1e-4) {
                    cerr << "ERROR: In column " << t << " row " << i << " should be "
                         << original[i] << " but is " << global_res[i] << "\n";
                    success = false;
                }
            }

            free(original);
            free(recvcounts);
            free(displs);
            free(global_res);
        }
    }

    free(v);
    free(res);

    int success_i = success ? 1 : 0;
    int global_success;
    MPI_Bcast(&success_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    global_success = success_i;

    if (rank == 0) {
        printf(global_success ? "RECONSTRUCT PASSED\n" : "RECONSTRUCT FAILED\n");
    }
    return global_success != 0;
}

// ---------------------------------------------------------------------------
// Correctness: round-trip a random vector through ILU_multiply then
// ILU_solve and check we recover the original. Cheap enough to run at any
// matrix size (cost is O(nnz + separator comm) per trial, not O(N)).
// ---------------------------------------------------------------------------
bool test_vector(struct ILUFact* ilu, int N, int n_trials, unsigned int seed)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int first_row = test_rank_first_row(rank, N, world_size);
    int last_row = test_rank_first_row(rank + 1, N, world_size);
    int n_local_rows = last_row - first_row;

    // Same seed on every rank -> identical RNG stream on every rank, so the
    // full-length vector v is identical across ranks each trial.
    std::mt19937 rng(seed);

    std::vector<double> v(N);
    double* x_true = (double*) malloc(n_local_rows * sizeof(double));
    double* b      = (double*) malloc(n_local_rows * sizeof(double));
    double* x_test = (double*) malloc(n_local_rows * sizeof(double));

    int overall_success = 1;

    for (int trial = 0; trial < n_trials; trial++)
    {
        for (int i = 0; i < N; i++) v[i] = (double) i;
        std::shuffle(v.begin(), v.end(), rng);

        memcpy(x_true, v.data() + first_row, n_local_rows * sizeof(double));

        if (rank == 0) { fprintf(stderr, "about to multiply\n"); fflush(stderr); }
        ILU_multiply(ilu, x_true, b);
        if (rank == 0) { fprintf(stderr, "multiply done, about to solve\n"); fflush(stderr); }
        ILU_solve(ilu, b, x_test);
        if (rank == 0) { fprintf(stderr, "solve done\n"); fflush(stderr); }

        int trial_success = 1;
        for (int i = 0; i < n_local_rows; i++)
        {
            if (abs(x_true[i] - x_test[i]) > EPS)
            {
                cout << "Trial: " << trial << " Rank: " << rank << " row: " << first_row + i
                << " expected: " << std::setprecision(15) << x_true[i]
                << " got: " << std::setprecision(15) << x_test[i] << "\n";
                trial_success = 0;
            }
        }

        int trial_passed;
        MPI_Reduce(&trial_success, &trial_passed, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            if (trial_passed == 0)
            {
                printf("TRIAL %d FAILED\n", trial);
                overall_success = 0;
            }
            else
            {
                printf("TRIAL %d PASSED\n", trial);
            }
        }
    }

    MPI_Bcast(&overall_success, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf(overall_success ? "TEST PASSED\n" : "TEST FAILED\n");
    }

    free(x_true);
    free(b);
    free(x_test);
    return overall_success != 0;
}

// ---------------------------------------------------------------------------
// Performance: time ILU_factorize in isolation. Prints one grep-able line
// from rank 0: PERF,phase=factorize,N=...,nnz=...,P=...,time=...
// ---------------------------------------------------------------------------
struct ILUFact* time_factorize(int N, int nnz, int* row, int* col, double* val,
                                int rank, int world_size, double* out_time)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    struct ILUFact* ilu = ILU_factorize(N, nnz, row, col, val);
    double t1 = MPI_Wtime();

    double local_time = t1 - t0, max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("PERF,phase=factorize,N=%d,nnz=%d,P=%d,time=%.6f\n",
               N, nnz, world_size, max_time);
        *out_time = max_time;
    }
    return ilu;
}

// ---------------------------------------------------------------------------
// Performance: time ILU_solve and ILU_multiply, averaged over n_reps calls
// on a fixed vector. Kept separate from correctness so nothing here
// contributes O(N) work beyond what the library itself does.
// ---------------------------------------------------------------------------
void time_solve_multiply(struct ILUFact* ilu, int N, int n_reps,
                          int rank, int world_size)
{
    int first_row = test_rank_first_row(rank, N, world_size);
    int last_row  = test_rank_first_row(rank + 1, N, world_size);
    int n_local   = last_row - first_row;

    std::mt19937 rng(1234);
    std::vector<double> x_local(n_local), b_local(n_local), tmp(n_local);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& xv : x_local) xv = dist(rng);

    // ILU_multiply timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < n_reps; i++) {
        ILU_multiply(ilu, x_local.data(), b_local.data());
    }
    double t1 = MPI_Wtime();
    double local_mult = (t1 - t0) / n_reps, max_mult;
    MPI_Reduce(&local_mult, &max_mult, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("PERF,phase=multiply,N=%d,P=%d,time=%.6f\n", N, world_size, max_mult);

    // ILU_solve timing (reuse b_local as rhs from the multiply above)
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    for (int i = 0; i < n_reps; i++) {
        ILU_solve(ilu, b_local.data(), tmp.data());
    }
    double t3 = MPI_Wtime();
    double local_solve = (t3 - t2) / n_reps, max_solve;
    MPI_Reduce(&local_solve, &max_solve, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("PERF,phase=solve,N=%d,P=%d,time=%.6f\n", N, world_size, max_solve);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <matrix.mtx> <mode> [options]\n"
        "\n"
        "Modes (can be combined, e.g. \"correctness,perf\"):\n"
        "  reconstruct   Full column-by-column reconstruction check.\n"
        "                O(N) communication rounds -- small/medium matrices only.\n"
        "  vector        Random vector round-trip check (multiply then solve).\n"
        "                Cheap; safe to run at any size.\n"
        "  perf          Time factorize (isolated) + solve/multiply (isolated).\n"
        "                No correctness checking, no O(N) loops.\n"
        "  correctness   Shorthand for \"reconstruct,vector\".\n"
        "  all            Shorthand for \"reconstruct,vector,perf\".\n"
        "\n"
        "Options:\n"
        "  --trials N        Number of trials for the vector test (default 3)\n"
        "  --perf-reps N     Number of repetitions for solve/multiply timing (default 20)\n"
        "  --seed N          RNG seed for the vector test (default 42)\n",
        prog);
}

bool has_mode(const string& modes, const string& name)
{
    return modes.find(name) != string::npos;
}

int main(int argc, char* argv[])
{
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    char* matrix_path = argv[1];
    string mode_arg = argv[2];
    if (mode_arg == "correctness") mode_arg = "reconstruct,vector";
    if (mode_arg == "all")         mode_arg = "reconstruct,vector,perf";

    int n_trials   = 3;
    int perf_reps  = 20;
    unsigned int seed = 42;

    for (int i = 3; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--trials"    && i + 1 < argc) n_trials  = atoi(argv[++i]);
        else if (arg == "--perf-reps" && i + 1 < argc) perf_reps = atoi(argv[++i]);
        else if (arg == "--seed"      && i + 1 < argc) seed      = (unsigned int) atoi(argv[++i]);
        else if (rank == 0) fprintf(stderr, "Warning: unrecognized option '%s'\n", arg.c_str());
    }

    // --- Load matrix on rank 0 only ---
    int N = 0, nnz = 0;
    int* row = NULL;
    int* col = NULL;
    double* val = NULL;
    if (rank == 0) {
        double t0 = MPI_Wtime();
        read_matrix(matrix_path, &N, &nnz, &row, &col, &val);
        double t1 = MPI_Wtime();
        printf("Loaded %s: N=%d, nnz=%d (read took %.3fs)\n",
               matrix_path, N, nnz, t1 - t0);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool any_correctness = has_mode(mode_arg, "reconstruct") || has_mode(mode_arg, "vector");
    bool run_perf         = has_mode(mode_arg, "perf");

    bool overall_ok = true;

    // --- Correctness path: factorize once (untimed), run requested checks ---
    if (any_correctness) {
        struct ILUFact* ilu = ILU_factorize(N, nnz, row, col, val);

        if (has_mode(mode_arg, "reconstruct")) {
            if (N > 20000) {
                log0(rank, "Skipping reconstruct: N=%d is too large for the O(N) "
                           "reconstruction check. Use a smaller matrix for this mode.\n", N);
            } else {
                overall_ok &= test_reconstruct(N, nnz, row, col, val, ilu);
            }
        }

        if (has_mode(mode_arg, "vector")) {
            overall_ok &= test_vector(ilu, N, n_trials, seed);
        }

        ILU_free(ilu);
    }

    // --- Performance path: separate, isolated timing, no correctness mixed in ---
    if (run_perf) {
        double factorize_time = 0.0;
        struct ILUFact* ilu_perf = time_factorize(N, nnz, row, col, val,
                                                   rank, world_size, &factorize_time);
        time_solve_multiply(ilu_perf, N, perf_reps, rank, world_size);
        ILU_free(ilu_perf);
    }

    if (rank == 0) free(row);
    if (rank == 0) free(col);
    if (rank == 0) free(val);

    if (rank == 0) {
        printf(overall_ok ? "OVERALL: PASSED\n" : "OVERALL: FAILED\n");
    }

    MPI_Finalize();
    return overall_ok ? 0 : 1;
}