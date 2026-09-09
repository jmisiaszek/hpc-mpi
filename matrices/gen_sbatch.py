import sys
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

def generate_test_matrix(N, density, filename):
    custom_random = lambda size: np.random.uniform(-100.0, 100.0, size=size)
    # The density parameter is now passed dynamically
    A = sp.random(N, N, density=density, format='csr', data_rvs=custom_random)
    row_sums = np.array(np.abs(A).sum(axis=1)).flatten()
    diagonal_boost = sp.diags(row_sums + 1.0)
    A_dominant = A + diagonal_boost
    
    sio.mmwrite(filename, A_dominant)

if __name__ == "__main__":
    # Now expecting 4 arguments: script_name, N, density, filename
    if len(sys.argv) != 4:
        print("Usage: python3 generate_matrix.py <N> <density> <output_file>")
        sys.exit(1)
        
    N = int(sys.argv[1])
    density = float(sys.argv[2])
    filename = sys.argv[3]
    
    generate_test_matrix(N, density, filename)