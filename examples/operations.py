import PyVSparse
import numpy as np
import scipy as sp

# Setup
csc_mat = sp.sparse.random(3, 3, format='csc', dtype=np.float32, density=1)

vcsc = PyVSparse.VCSC(csc_mat)
ivcsc = PyVSparse.IVCSC(csc_mat)

# Both formats use the same syntax for operations
# Scalar Multiplication

vcsc *= 2 
# or 
vcsc = vcsc * 2 # But this is more expensive because it creates a new matrix

ivcsc *= 2
# or
ivcsc = ivcsc * 2

# Vector Multiplication
# Because Sparse * Dense operations produce dense objects, the result will be a numpy array.
# Sparse * Sparse operations are not supported.

vector = np.array([1, 2, 3], dtype=np.float32)

dense_numpy_array1 = vcsc * vector
dense_numpy_array2 = ivcsc * vector

# Matrix Multiplication
# Like vector multiplication, Sparse * Sparse operations are not supported.
# SpMM is supported and returns a dense numpy array.
# The * operator is used for SpMM, unlike the * operator in scipy (which uses @).
dense_numpy_array3 = vcsc * csc_mat.toarray()
dense_numpy_array4 = ivcsc * csc_mat.toarray()


# Transpose
# Transpose is supported for both VCSC and IVCSC matrices.
# Unfortunately, transpose is very slow becasue of how the data is stored.

ivcsc_transpose = ivcsc.transpose()
# or an in place transpose
ivcsc.transpose(inplace=True)


# Equality
print(vcsc == ivcsc) # NotImplementedError
print(vcsc == vcsc) # True
print(ivcsc == ivcsc) # True

# Inequality
print(vcsc != ivcsc) # NotImplementedError
print(vcsc != vcsc) # False
print(ivcsc != ivcsc) # False

# Random Access
print(vcsc[0, 0]) 

# Max
print(vcsc.max()) 
print(vcsc.max(axis=0)) # Max of each column
print(vcsc.max(axis=1)) # Max of each row

# Min
# Note: Because of the way the matrix is stored, minimums that are zero are very expensive to compute.
# The returned value is the minimum non-zero value unless all values are zero or nnz < rows*cols (for min(None))
print(vcsc.min())
print(vcsc.min(axis=0)) # Min of each column
print(vcsc.min(axis=1)) # Min of each row

# Sum
print(vcsc.sum())
print(vcsc.sum(axis=0)) # Sum of each column
print(vcsc.sum(axis=1)) # Sum of each row

# Trace
print(vcsc.trace())

