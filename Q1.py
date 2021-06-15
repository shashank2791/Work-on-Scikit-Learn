import numpy as np
from scipy import sparse

mat = np.eye(4)
print("NumPy array:\n", mat)
sparse_matrix = sparse.csr_matrix(mat)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)