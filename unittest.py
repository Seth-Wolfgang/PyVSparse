
import PyVSparse as pvs
import scipy as sp
import numpy as np


control = sp.sparse.random(100, 100, format='csc', dtype = np.int8)
test = pvs.VCSC__int32_t_uint32_t_Col(control)
test2 = pvs.IVCSC__int32_t_int32_t_Col(control)

assert control.sum() == test.sum()
assert control.sum() == test2.sum()

