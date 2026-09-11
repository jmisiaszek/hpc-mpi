# HPC 2 -- MPI DAS-ILU algorithm

## 1. Algorithm description and main differences

My implementation of the algorithm follows mostly the explanation provided in task description.

At the start rows are split equally between all mpi processes.
This is also a difference from the original DAS-ILU, which performs domain decomposition with graph partitioning.

Rows are split within each process into interior and separator rows based on dependencies on lower ranks.
Then all interior rows are factorized once, because all necessary information is already available for the process.

Separator rows are factorized in a while loop. In each loop first all processes exchange information needed for factorization, and then calculate new values for separator rows. This means that the algorithm uses dependencies from the previous loop. This does not break the correctness but needs few more iterations.

Main difference between my code and the original DAS-ILU is that my code synchronizes at the very end of the loop with `MPI_Allreduce` to calculate the maximal difference, while the algorithm in the paper is completely asynchronous.

## 2. Tests and performance

I tested my implementation using randomly generated sparse matrices with sizes 10, 100 and 1000 and densities of 10%, 20% and 30%. I also tested the program for 1-10 worker counts. For the tests themselves, I did 2 different calculations:

1. **Matrix reconstruction**. After calculating `ILUFact` for a given matrix, I multiplied it by vectors from standard basis. Then I checked if the results match vectors from the original matrix.

2. **Multiplication Identity**. The first tests checks correctness of `ILU_factorize` and `ILU_multiply`. This test checks for `ILU_solve`. I generated n random vectors and multiplied those vectors by the ILU matrix using `ILU_multiply`. Then I ran those results through `ILU_solve` and verifed that the results found matched the original vector.

Here is the average runtime for all configurations. Each configuration was run on 100 different matrices.

![](./scaling_results_plot.png)

### 1m2d.mtx  (N=1000000, nnz=4996000)

| P | factorize (s) | speedup | eff | multiply (s) | speedup | eff | solve (s) | speedup | eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6302 | 1.00x | 100% | 2.0530 | 1.00x | 100% | 0.0352 | 1.00x | 100% |
| 4 | 0.2900 | 2.17x | 54% | 0.5086 | 4.04x | 101% | 0.0183 | 1.92x | 48% |
| 8 | 0.2481 | 2.54x | 32% | 0.2274 | 9.03x | 113% | 0.0178 | 1.98x | 25% |
| 12 | 0.2203 | 2.86x | 24% | 0.1212 | 16.94x | 141% | 0.0186 | 1.90x | 16% |
| 16 | 0.2247 | 2.80x | 18% | 0.0984 | 20.86x | 130% | 0.0195 | 1.81x | 11% |
| 20 | 0.2017 | 3.12x | 16% | 0.0703 | 29.22x | 146% | 0.0206 | 1.71x | 9% |
| 24 | 0.2031 | 3.10x | 13% | 0.0702 | 29.24x | 122% | 0.0219 | 1.61x | 7% |
| 28 | 0.2192 | 2.88x | 10% | 0.0622 | 33.02x | 118% | 0.0265 | 1.33x | 5% |
| 32 | 0.2724 | 2.31x | 7% | 0.0517 | 39.74x | 124% | 0.0305 | 1.15x | 4% |
| 36 | 0.2773 | 2.27x | 6% | 0.0476 | 43.10x | 120% | 0.0344 | 1.02x | 3% |
| 40 | 0.2502 | 2.52x | 6% | 0.0389 | 52.73x | 132% | 0.0378 | 0.93x | 2% |
| 44 | 0.2542 ~2 | 2.48x | 6% | 0.0354 | 57.98x | 132% | 0.0410 | 0.86x | 2% |
| 48 | 0.2633 | 2.39x | 5% | 0.0311 | 66.09x | 138% | 0.0445 | 0.79x | 2% |

Medians across repeats; speedup and efficiency relative to P=1. `~N` marks a cell with only N repeat(s).

### 1m3d.mtx  (N=1000000, nnz=6940000)

| P | factorize (s) | speedup | eff | multiply (s) | speedup | eff | solve (s) | speedup | eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8413 | 1.00x | 100% | 2.7568 | 1.00x | 100% | 0.0459 | 1.00x | 100% |
| 4 | 0.4074 | 2.06x | 52% | 0.9128 | 3.02x | 76% | 0.0360 | 1.28x | 32% |
| 8 | 0.3487 | 2.41x | 30% | 0.2334 | 11.81x | 148% | 0.0532 | 0.86x | 11% |
| 12 | 0.3473 | 2.42x | 20% | 0.1561 | 17.67x | 147% | 0.0723 | 0.63x | 5% |
| 16 | 0.3075 | 2.74x | 17% | 0.1099 | 25.09x | 157% | 0.0933 | 0.49x | 3% |
| 20 | 0.3071 | 2.74x | 14% | 0.0850 | 32.42x | 162% | 0.1112 | 0.41x | 2% |
| 24 | 0.3158 | 2.66x | 11% | 0.0722 | 38.18x | 159% | 0.1331 | 0.35x | 1% |
| 28 | 0.3012 | 2.79x | 10% | 0.0794 | 34.70x | 124% | 0.1676 | 0.27x | 1% |
| 32 | 0.3131 | 2.69x | 8% | 0.0720 | 38.28x | 120% | 0.2023 | 0.23x | 1% |
| 36 | 0.3198 | 2.63x | 7% | 0.0659 | 41.85x | 116% | 0.2393 | 0.19x | 1% |
| 40 | 0.3924 | 2.14x | 5% | 0.0613 | 44.97x | 112% | 0.2691 | 0.17x | 0% |
| 44 | 0.3965 | 2.12x | 5% | 0.0633 | 43.59x | 99% | 0.3069 | 0.15x | 0% |
| 48 | 0.4153 | 2.03x | 4% | 0.0561 | 49.14x | 102% | 0.3418 | 0.13x | 0% |

Medians across repeats; speedup and efficiency relative to P=1. `~N` marks a cell with only N repeat(s).


### P = 16

| max offset | factorize (s) | multiply (s) | solve (s) |
|---|---:|---:|---:|
| 4 | 0.3144 | 0.1106 | 0.0189 |
| 16 | 0.3200 | 0.0920 | 0.0190 |
| 64 | 0.3139 | 0.1073 | 0.0191 |
| 256 | 0.3172 | 0.1185 | 0.0197 |
| 1024 | 0.3036 | 0.1205 | 0.0236 |
| 4096 | 0.2889 | 0.0964 | 0.0435 |
| 16384 | 0.3005 | 0.1207 | 0.1543 |
| 65536 | 0.9599 | 0.2896 | 0.6933 |
| 131072 | 0.7189 | 0.2659 | 0.5019 |

Medians across repeats. `~N` marks a cell with only N repeat(s).