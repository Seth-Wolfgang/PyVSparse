

from vcsc import VCSC
from ivcsc import IVCSC
import scipy as sp
import numpy as np
from scipy import sparse



control = sp.sparse.random(100, 100, format='csc', dtype = np.int32)
test = VCSC(control)
test2 = IVCSC(control)
print(type(test.wrappedForm))
print(control.format)


assert control.sum() == test.sum() and control.sum() == test2.sum() , ( 
    "sum" + "SciPy: " + str(control.sum()) + " PyVSparse: " + str(test.sum()))

assert control.trace() == test.trace(), (
    "trace" + "SciPy: " + str(control.trace()) + " PyVSparse: " + str(test.trace()))

assert control.outer.sum() == test.outerSum(), (
    "outerSum" + "SciPy: " + str(control.outer.sum()) + " PyVSparse: " + str(test.outerSum()))
assert control.inner.sum() == test.innerSum(), (
    "innerSum" + "SciPy: " + str(control.inner.sum()) + " PyVSparse: " + str(test.innerSum()))
assert control == test.toSciPySparse(), "toSciPySparse"
assert control.transpose() == test.transpose().toSciPySparse(), "transpose"
