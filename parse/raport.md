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

