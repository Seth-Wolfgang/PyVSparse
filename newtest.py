

from curses.ascii import SP
from vcsc import VCSC
from ivcsc import IVCSC
import scipy as sp
import numpy as np
# from scipy import sparse
# import vcsc
import PyVSparse
# print(dir(VCSC))
# print(dir(IVCSC))
# print(dir(PyVSparse.IVCSC._uint32_Col))


control: sp.sparse.csc_matrix = sp.sparse.random(5, 1, format='csc', dtype = np.int32, density=1)
control2 = sp.sparse.random(5, 5, format='csr', dtype = np.int8, density=1)
control3 = np.ones((1, 1), dtype = np.int8)

vcsctest = VCSC(control2)
ivcsctest = IVCSC(control2)




IVCSCresult = ivcsctest * control3
VCSCresult = vcsctest * control3

print(type(IVCSCresult))
print(type(VCSCresult))

print("IVCSC: \n", IVCSCresult) 
print("VCSC: \n", VCSCresult)
SPresult = control2 * control3

print(type(SPresult))

print("SciPy: \n",SPresult)
print("Sub:\n " ,  SPresult - IVCSCresult)













# print(test)
# print(test2)

# vcscResult = test * control2.todense()
# ivcscResult = test2 * control2.todense()
# cscResult = control2 * control2.todense()

# print(control.todense())
# print("SciPy")
# print(control2.todense())

# print("VCSC: \n")
# print(vcscResult)
# print("IVCSC: \n")
# print(ivcscResult)
# print("CSC: \n")
# print(cscResult)

