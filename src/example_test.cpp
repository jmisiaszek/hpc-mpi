#include <cstdio>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "ilu.h"

using namespace std;

#define N_TESTS 10

#define EPS 10e-4

int test_rank_first_row(int rank, int N, int world_size)
{
    return rank * N / world_size;
}

// Reads matrix in a MatrixMarket format
void read_matrix(char* in_file, int* N, int* nnz, int** row, int** col, double** val)
{
    FILE *fp = fopen(in_file, "r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int l = -1;
    if (fp == NULL)
        exit(EXIT_FAILURE);

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

// bool test_vector(struct ILUFact* ilu, int N, double* v)
// {
//     int rank, world_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     int first_row = test_rank_first_row(rank, N, world_size);
//     int last_row = test_rank_first_row(rank + 1, N, world_size);
//     int n_local_rows = last_row - first_row;

//     int success = 1;

//     // x_true is your known pattern
//     double* x_true = (double*) malloc(n_local_rows * sizeof(double));
//     memcpy(x_true, v + first_row, n_local_rows * sizeof(double));
    
//     double* b = (double*) malloc(n_local_rows * sizeof(double));
//     double* x_test = (double*) malloc(n_local_rows * sizeof(double));
    
//     // 1. GENERATE THE RHS (b = LU * x_true)
//     ILU_multiply(ilu, x_true, b);

//     // 2. RUN THE SOLVER (Solve for x_test using b)
//     ILU_solve(ilu, b, x_test);

//     // 3. COMPARE
//     for (int i = 0; i < n_local_rows; i++)
//     {
//         if (abs(x_true[i] - x_test[i]) > EPS)
//         {
//             cout << "Rank: " << rank << " row: " << first_row + i 
//                  << " expected: " << x_true[i] << " got: " << x_test[i] << "\n";
//             success = 0;
//         }
//     }

//     int passed;
//     MPI_Reduce(&success, &passed, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
//     if (rank == 0) {
//         if(passed == 0) printf("TEST FAILED\n");
//         else            printf("TEST PASSED\n");
//     }

//     free(x_true);
//     free(b);
//     free(x_test);
//     return passed;
// }

bool test_vector(struct ILUFact* ilu, int N, int n_trials, unsigned int seed)
{
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int first_row = test_rank_first_row(rank, N, world_size);
    int last_row = test_rank_first_row(rank + 1, N, world_size);
    int n_local_rows = last_row - first_row;

    // Same seed on every rank -> identical RNG stream on every rank,
    // so the full-length vector v is identical across ranks each trial.
    std::mt19937 rng(seed);

    std::vector<double> v(N);
    double* x_true = (double*) malloc(n_local_rows * sizeof(double));
    double* b      = (double*) malloc(n_local_rows * sizeof(double));
    double* x_test = (double*) malloc(n_local_rows * sizeof(double));

    int overall_success = 1;

    for (int trial = 0; trial < n_trials; trial++)
    {
        // Build a fresh random permutation of 0..N-1 as the test pattern
        for (int i = 0; i < N; i++) v[i] = (double) i;
        std::shuffle(v.begin(), v.end(), rng);

        memcpy(x_true, v.data() + first_row, n_local_rows * sizeof(double));

        // 1. Generate RHS: b = LU * x_true
        ILU_multiply(ilu, x_true, b);

        // 2. Solve for x_test using b
        ILU_solve(ilu, b, x_test);

        // 3. Compare
        int trial_success = 1;
        for (int i = 0; i < n_local_rows; i++)
        {
            if (abs(x_true[i] - x_test[i]) > EPS)
            {
                cout << "Trial: " << trial << " Rank: " << rank << " row: " << first_row + i
                     << " expected: " << x_true[i] << " got: " << x_test[i] << "\n";
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

    // Make sure every rank agrees on the final result (rank 0 is authoritative here)
    MPI_Bcast(&overall_success, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (overall_success == 0) printf("TEST FAILED\n");
        else                      printf("TEST PASSED\n");
    }

    free(x_true);
    free(b);
    free(x_test);
    return overall_success;
}

void test_reconstruct(int N, int nnz, int* row, int* col, double* val, struct ILUFact* ilu) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int first_row = test_rank_first_row(rank, N, world_size);
    int last_row = test_rank_first_row(rank + 1, N, world_size);
    int n_local_rows = last_row - first_row;

    bool success = 1;
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
                    cerr << "ERROR: In column " << t << " row " << i << " should be " << original[i] << " but is " << global_res[i] << "\n"; 
                    success = 0;
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
    if (rank == 0) {
        if(success == 0)
        {
            printf("RECONSTRUCT FAILED\n");
        }
        else
        {
            printf("RECONSTRUCT PASSED\n");
        }
    }
}

int main(int argc, char* argv[])
{
    assert(argc == 3);

    int rank;
    int world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 0, nnz = 0;
    int* row = NULL;
    int* col = NULL;
    double* val = NULL;
    if (rank == 0)
    {
        read_matrix(argv[1], &N, &nnz, &row, &col, &val);
    }

    struct ILUFact* ilu;
    ilu = ILU_factorize(N, nnz, row, col, val);

    // cerr << "main " << rank << " " << row << " " << col << " " << val << "\n";
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (string(argv[2]) == "1") {
        test_reconstruct(N, nnz, row, col, val, ilu);
    }

    free(row);
    free(col);
    free(val);

    if (string(argv[2]) == "2") {
        unsigned int seed = 42; // fixed seed -> reproducible "random" trials
        test_vector(ilu, N, N_TESTS, seed);
    }

    ILU_free(ilu);
    MPI_Finalize();
    return 0;
}
