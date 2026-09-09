import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import subprocess
import os
import sys
import time

# Configuration
EXECUTABLE = "./example_test" # Change this to your compiled binary name
NUM_TESTS = 2
N_SIZE = 10
MPI_PROCS = 2

reconstruct_times = []
vector_times = []

def generate_test_matrix(N, filename):
    # 1. Create a random sparse matrix (10% density)
    # A = sp.random(N, N, density=0.1, format='csr', data_rvs=np.random.randn)
    custom_random = lambda size: np.random.uniform(-100.0, 100.0, size=size)
    A = sp.random(N, N, density=0.1, format='csr', data_rvs=custom_random)

    # 2. Make it strictly diagonally dominant to prevent 0-pivots during ILU
    # Sum the absolute values of the rows and add it to the diagonal
    row_sums = np.array(np.abs(A).sum(axis=1)).flatten()
    diagonal_boost = sp.diags(row_sums + 1.0)
    
    A_dominant = A + diagonal_boost
    
    # 3. Save as MatrixMarket format (scipy automatically uses 1-based indexing!)
    sio.mmwrite(filename, A_dominant)

def run_test(proc_count, test_id):
    mtx_file = f"test.mtx"
    generate_test_matrix(N_SIZE, mtx_file)

    # reconstruct test
    cmd = ["mpiexec", "-n", str(proc_count), EXECUTABLE, mtx_file, '1']
    
    # Run the C++ program and capture the output
    start = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    end = time.time()
    reconstruct_times.append(end - start)

    # Clean up the matrix file to save disk space
    # os.remove(mtx_file)
    
    # Check if the program failed or crashed
    if result.returncode != 0:
        print(f"Reconstruction test {test_id} CRASHED! Return code: {result.returncode}")
        print(result.stderr)
        return False
        
    if "FAILED" in result.stdout:
        print(f"Reconstruction test {test_id} FAILED LOGIC CHECK!")
        print(result.stdout)
        return False

    # vector tests
    cmd = ["mpiexec", "-n", str(proc_count), EXECUTABLE, mtx_file, '2']
        
    # Run the C++ program and capture the output
    start = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    end = time.time()
    vector_times.append(end - start)

    # Clean up the matrix file to save disk space
    # os.remove(mtx_file)
    
    # Check if the program failed or crashed
    if result.returncode != 0:
        print(f"Vector test {test_id} CRASHED! Return code: {result.returncode}")
        print(result.stderr)
        return False
        
    if "FAILED" in result.stdout:
        print(f"Vector test {test_id} FAILED LOGIC CHECK!")
        print(result.stdout)
        return False
        
    return True

for p in range(1, MPI_PROCS + 1):

    print(f"Starting {NUM_TESTS} automated tests with N={N_SIZE} on {p} procs...")

    passed_count = 0

    for i in range(NUM_TESTS):
        if run_test(p, i):
            passed_count += 1
            print(f"Test {i+1}/{NUM_TESTS} Passed.", end='\r')
        else:
            print(f"\nTesting stopped due to failure on test {i}.")
            sys.exit(1)

    print(f"\nAll {passed_count}/{NUM_TESTS} tests PASSED!")
    print(f"Average reconstruction runtime: {np.mean(reconstruct_times)}")
    print(f"Average vecotr runtime: {np.mean(vector_times)}")
