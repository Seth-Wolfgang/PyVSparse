
import PyVSparse
import numpy as np
import scipy as sp

from PyVSparse import ivcsc


# Because PyVSparse is for sparse matrices, we need to use scipy.sparse formats to create the matrices
# Currently, only COO, CSC, and CSR are supported becasue the C++ backend uses the Eigen library.

# Create a 3x3 sparse matrix. Any dtype works as long as its a C++ supported type
csc_mat = sp.sparse.random(3, 3, format='csc', dtype=np.float32, density=1)
csr_mat = sp.sparse.random(3, 3, format='csr', dtype=np.int32, density=1)
coo_mat = sp.sparse.random(3, 3, format='coo', dtype=np.float64, density=1)
# Note: There must be at least 1 nonzero value in the matrix.

# Create a PyVSparse matrix from the CSC matrix
vcsc_mat1 = PyVSparse.VCSC(csc_mat)
ivcsc_mat1 = PyVSparse.IVCSC(csc_mat)

# Create a PyVSparse matrix from the CSR matrix
vcsr_mat2 = PyVSparse.VCSC(csr_mat)
ivcsc_mat2 = PyVSparse.IVCSC(csr_mat)

# Create a PyVSparse matrix from the COO matrix
vcsc_mat3 = PyVSparse.VCSC(coo_mat)
ivcsc_mat3 = PyVSparse.IVCSC(coo_mat)

# And this can be converted back into a CSC or CSR matrix. tocoo() is not (yet) supported
csc_mat1 = vcsc_mat1.tocsc()
csr_mat1 = ivcsc_mat1.tocsr() # It does not matter how it was created

print(csc_mat1 == csc_mat)


## VCSC and Index Types

# VCSC will store row/column indicies as 4 byte unsigned integers by default
# But this can be adjusted during construction.

# Create a VCSC matrix with 2 byte unsigned integers
vcsc_mat4 = PyVSparse.VCSC(csc_mat, indexType=np.uint16)

# This is not available for IVCSC matrices because they will use the smallest 
# possible index type to for every value.

## Row/Column Major

# VCSC and IVCSC matrices can be created in row major format as well.
vcsc_mat5 = PyVSparse.VCSC(csc_mat, order="row") # "Row" or "Col", default is "Col". Capitalization does not matter
ivcsc_mat5 = PyVSparse.IVCSC(csc_mat, order="row")

# Note: By default, a csr matrix will create a row major I/VCSC matrix.

# To combine everything,
vcsc_mat6 = PyVSparse.VCSC(csc_mat, indexType=np.uint8, order="row") # is a valid matrix
ivcsc_mat6 = PyVSparse.IVCSC(csc_mat, order="row") # is a valid matrix

# it is worth mentioning, I/VCSC rely on other matrices to be created. 
# It is difficult, and expensive, to write to this format, so matrices are read from other formats.