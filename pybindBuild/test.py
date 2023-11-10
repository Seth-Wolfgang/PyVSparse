import PyVSparse
import scipy as sp
import numpy as np


print("hello world")

print(dir(PyVSparse))
# print(dir(PyVSparse.VCSC_i_h_2_Col))

scipySparseMat = sp.sparse.random(10000, 10000, format='csc', dtype = np.int8)
print(scipySparseMat.indices.dtype)
print(scipySparseMat.data.dtype)

scipySparseMat[:] = 1

test3 = PyVSparse.IVCSC__int32_t_int32_t_Col(scipySparseMat)
test2 = PyVSparse.VCSC__int32_t_uint32_t_Col(scipySparseMat)


# test.print()
print("IVCSC", test3.sum())
print("VCSC", test2.sum())
print("scipy", scipySparseMat.sum())
