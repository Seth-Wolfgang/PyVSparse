import PyVSparse
import numpy as np
import scipy as sp

# File I/O is an important way to keep your data.
# PyVSparse supports reading/writing files in our own format.

csc_mat = sp.sparse.random(3, 3, format='csc', dtype=np.float32, density=1)

# Write the matrix to a file
vcsc = PyVSparse.VCSC(csc_mat)
vcsc.write('vcsc_mat') # This will automatically append .vcsc to the end, if not already there

# Read the matrix from the file
vcsc2 = PyVSparse.VCSC('vcsc_mat.vcsc')

# Importantly, the written file will maintain the information of the original matrix.
# So if the written matrix uses 2 byte index types, the read matrix will also use 2 byte index types.
# No customization or extra construction paramters are needed.

vcsc2 = PyVSparse.VCSC(csc_mat, indexType=np.uint16)
vcsc2.write('vcsc_mat2') 

vcsc3 = PyVSparse.VCSC('vcsc_mat2.vcsc')
print(vcsc3.indexType == np.uint16) # True


# The same can be done with IVCSC matrices, however, IVCSC matrices do not have index type parameters.
ivcsc = PyVSparse.IVCSC(csc_mat)
ivcsc.write('ivcsc_mat')

ivcsc2 = PyVSparse.IVCSC('ivcsc_mat.ivcsc')
print(ivcsc2 == ivcsc2) # True


# PyVSparse also supports reading/writing files in the .npz format.
# However, this is essentially a wrapper around scipy's save_npz and load_npz functions.
import os
try:
    sp.sparse.save_npz("test.npz", csc_mat)
    newVCSC = PyVSparse.VCSC("test.npz")
finally:
    os.remove("test.npz")
print(newVCSC.tocsc().toarray() == csc_mat.toarray()) # True

try:
    sp.sparse.save_npz("test.npz", csc_mat)
    newIVCSC = PyVSparse.IVCSC("test.npz")
finally:
    os.remove("test.npz")
print(newIVCSC.tocsc().toarray() == csc_mat.toarray()) # true
