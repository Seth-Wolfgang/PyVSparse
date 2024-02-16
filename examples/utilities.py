import PyVSparse
import numpy as np
import scipy as sp

# Setup
csc_mat = sp.sparse.random(4, 4, format='csc', dtype=np.int32, density=1)
print(csc_mat.nnz)

vcsc = PyVSparse.VCSC(csc_mat)
ivcsc = PyVSparse.IVCSC(csc_mat)


# Slice
# Slicing is supported for both VCSC and IVCSC matrices, but only along the major axis.
# Slicing along the minor axis is not supported.

vcsc_half = vcsc.slice(0, 2)
ivcsc_half = ivcsc.slice(0, 2)

# Appending
# Appending is supported for both VCSC and IVCSC matrices, but only along the major axis.
# CSC/CSR matrices can be appended to I/VCSC 
vcsc_appended = vcsc.append(csc_mat)
ivcsc_appended = ivcsc.append(csc_mat)
ivcsc_appended = ivcsc.append(ivcsc_appended)

# VCSC Indices, Values, and Counts
# VCSC matrices store their indices, values, and counts in separate arrays.
# These can be copied to a list

index = 1 # The major index to access i.e. column or row to get
indices = vcsc.getValues(index)
values = vcsc.getIndices(index)
counts = vcsc.getCounts(index)
numIndices = vcsc.getNumIndices(index)

# IVCSC does not support this becasue of how its stored. It would be useless.

# Shape 
print(vcsc.shape())
print(ivcsc.shape()[1])

# Bytesize
print(vcsc.byteSize())

