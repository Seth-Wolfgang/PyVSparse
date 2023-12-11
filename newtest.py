

# from vcsc import VCSC
# from ivcsc import IVCSC
# import scipy as sp
import numpy as np
# from scipy import sparse
# import vcsc
import PyVSparse
# print(dir(VCSC))
# print(dir(IVCSC))
print(dir(PyVSparse.IVCSC._uint32_Col))


# control: sp.sparse.csc_matrix = sp.sparse.random(5, 1, format='csc', dtype = np.int32, density=1)
# control2 = sp.sparse.random(5, 5, format='csc', dtype = np.int8, density=1)
# control3 = np.ones((5, 1), dtype = np.int8)
# test = VCSC(control2, indexT = np.uint8)
# test2 = IVCSC(control2)
# test4 = test * control3
# test3 = test2 * control3
# test5 = control2 * control3

# print(test4)
# print()
# print(test3)
# print()
# print(test5)

# assert control.sum() == test.sum() and control.sum() == test2.sum() , ( 
#     "sum" + "SciPy: " + str(control.sum()) + " PyVSparse: " + str(test.sum()))

# assert control.trace() == test.trace(), (
#     "trace" + "SciPy: " + str(control.trace()) + " PyVSparse: " + str(test.trace()))

# assert control.outer.sum() == test.outerSum(), (
#     "outerSum" + "SciPy: " + str(control.outer.sum()) + " PyVSparse: " + str(test.outerSum()))
# assert control.inner.sum() == test.innerSum(), (
#     "innerSum" + "SciPy: " + str(control.inner.sum()) + " PyVSparse: " + str(test.innerSum()))
# assert control == test.toSciPySparse(), "toSciPySparse"
# assert control.transpose() == test.transpose().toSciPySparse(), "transpose"
