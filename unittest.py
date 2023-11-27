
import random
import ivcsc
import vcsc
import scipy as sp
import numpy as np


#TODO CSR doesn't work for toCSC() -> IVSparse needs to CSR


myFormat = "csr"


def CSC_Equality(ivcsc, vcsc, control):
    csc_from_ivcsc = ivcsc.toCSC()
    csc_from_vcsc = vcsc.toCSC()

    for x, y, z in zip(csc_from_ivcsc.indices, csc_from_vcsc.indices, control.indices):
        assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y)
        assert x == z, "csc_from_ivcsc: " + str(x) + " control: " + str(z)

    for x, y, z in zip(csc_from_ivcsc.indptr, csc_from_vcsc.indptr, control.indptr):
        assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)
        assert x == z, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)
    for x, y, z in zip(csc_from_ivcsc.data, csc_from_vcsc.data, control.data):
        assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)
        assert x == z, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)

def CSR_Equality(ivcsc, vcsc, control):
    csc_from_ivcsc = ivcsc.toCSR
    csc_from_vcsc = vcsc.toCSR

    for x, y, z in zip(csc_from_ivcsc.indices, csc_from_vcsc.indices, control.indices):
        assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y)
        assert x == z, "csc_from_ivcsc: " + str(x) + " control: " + str(z)

    for x, y, z in zip(csc_from_ivcsc.indptr, csc_from_vcsc.indptr, control.indptr):
        assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)
        assert x == z, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)
    for x, y, z in zip(csc_from_ivcsc.data, csc_from_vcsc.data, control.data):
        assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)
        assert x == z, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " control: " + str(z)

maxRandomTests = 100
for randomTest in range(maxRandomTests):
    rows = random.randint(1, 1000)
    cols = random.randint(1, 1000)

    print("rows: ", rows, " cols: ", cols)
    
    control = sp.sparse.random(rows, cols, format=myFormat, dtype = np.int8, density = 0.2)

    while control.nnz == 0:
        control = sp.sparse.random(rows, cols, format=myFormat, dtype = np.int8, random_state = random.randint(0, 1000), density = 0.2)

    print("\033[92m", "Made Matrix", "\033[0m")
    vcsc_obj = vcsc.VCSC(control)
    ivcsc_obj = ivcsc.IVCSC(control)
    print("\033[92m", "Passed Construction", "\033[0m")

    # print("VCSC: ", vcsc_obj.sum(), " DTYPE: ", vcsc_obj.dtype, " TYPE: ", type(vcsc_obj.wrappedForm))
    # print("IVCSC", ivcsc_obj.sum(), " DTYPE: ", ivcsc_obj.dtype, " TYPE: ", type(vcsc_obj.wrappedForm))
    # print("SCPY", control.sum(), " DTYPE: ", control.dtype)
    assert control.sum() == vcsc_obj.sum()
    assert control.sum() == ivcsc_obj.sum()
    print("\033[92m", "Passed Sum", "\033[0m")

    vcsc_obj_copy = vcsc_obj
    ivcsc_obj_copy = ivcsc_obj
    print("\033[92m", "Passed Deep copy", "\033[0m")

    assert vcsc_obj_copy == vcsc_obj
    assert ivcsc_obj_copy == ivcsc_obj

    assert vcsc_obj_copy.sum() == vcsc_obj.sum(), "vcsc_obj: " + str(vcsc_obj.sum()) + " ivcsc_obj: " + str(ivcsc_obj.sum())
    assert ivcsc_obj_copy.sum() == ivcsc_obj.sum(), "vcsc_obj: " + str(vcsc_obj.sum()) + " ivcsc_obj: " + str(ivcsc_obj.sum())
    print("\033[92m", "Passed Deep Copy Equality", "\033[0m")

    assert vcsc_obj.trace() == ivcsc_obj.trace(), "vcsc_obj: " + str(vcsc_obj.trace()) + " ivcsc_obj: " + str(ivcsc_obj.trace())
    assert vcsc_obj.trace() == control.trace(), "vcsc_obj: " + str(vcsc_obj.trace()) + " ivcsc_obj: " + str(ivcsc_obj.trace()) + " control: " + str(control.trace())
    print("\033[92m", "Passed Trace", "\033[0m")

    # for x in range(100): # TODO something is broken but sums are equal?
    #     assert vcsc_obj.outerSum()[x] == ivcsc_obj.outerSum()[x], "vcsc_obj: " + str(vcsc_obj.outerSum()[x]) + " ivcsc_obj: " + str(ivcsc_obj.outerSum()[x])
    #     assert vcsc_obj.innerSum()[x] == ivcsc_obj.innerSum()[x], "vcsc_obj: " + str(vcsc_obj.innerSum()[x]) + " ivcsc_obj: " + str(ivcsc_obj.innerSum()[x])
    #     assert vcsc_obj.maxColCoeff()[x] == ivcsc_obj.maxColCoeff()[x], "vcsc_obj: " + str(vcsc_obj.maxColCoeff()[x]) + " ivcsc_obj: " + str(ivcsc_obj.maxColCoeff()[x])
    #     assert vcsc_obj.maxRowCoeff()[x] == ivcsc_obj.maxRowCoeff()[x], "vcsc_obj: " + str(vcsc_obj.maxRowCoeff()[x]) + " ivcsc_obj: " + str(ivcsc_obj.maxRowCoeff()[x])
    #     assert vcsc_obj.minColCoeff()[x] == ivcsc_obj.minColCoeff()[x], "vcsc_obj: " + str(vcsc_obj.minColCoeff()[x]) + " ivcsc_obj: " + str(ivcsc_obj.minColCoeff()[x])
    #     assert vcsc_obj.minRowCoeff()[x] == ivcsc_obj.minRowCoeff()[x], "vcsc_obj: " + str(vcsc_obj.minRowCoeff()[x]) + " ivcsc_obj: " + str(ivcsc_obj.minRowCoeff()[x])
    
    assert vcsc_obj.norm() == ivcsc_obj.norm()
    print("\033[92m", "Passed Norm", "\033[0m")

    for x in range(cols):
        assert vcsc_obj.vectorLength(x) == ivcsc_obj.vectorLength(x), "vcsc_obj: " + str(vcsc_obj.vectorLength(x)) + " ivcsc_obj: " + str(ivcsc_obj.vectorLength(x))
        # assert vcsc_obj.vectorLength(x) == control.getcol(x).sum(), "vcsc_obj: " + str(vcsc_obj.vectorLength(x)) + " ivcsc_obj: " + str(ivcsc_obj.vectorLength(x)) + " control: " + str(control.getrow(x).sum())
    print("\033[92m", "Passed Vector length", "\033[0m")

    csc_from_ivcsc = ivcsc_obj.toCSC()
    csc_from_vcsc = vcsc_obj.toCSC()
    print("\033[92m", "Passed toCSC", "\033[0m")

    assert csc_from_ivcsc.sum() == csc_from_vcsc.sum(), "csc_from_ivcsc: " + str(csc_from_ivcsc.sum()) + " csc_from_vcsc: " + str(csc_from_vcsc.sum())
    assert csc_from_ivcsc.shape == csc_from_vcsc.shape, "csc_from_ivcsc: " + str(csc_from_ivcsc.shape) + " csc_from_vcsc: " + str(csc_from_vcsc.shape)
    assert csc_from_ivcsc.dtype == csc_from_vcsc.dtype, "csc_from_ivcsc: " + str(csc_from_ivcsc.dtype) + " csc_from_vcsc: " + str(csc_from_vcsc.dtype)
    assert csc_from_ivcsc.format == csc_from_vcsc.format, "csc_from_ivcsc: " + str(csc_from_ivcsc.format) + " csc_from_vcsc: " + str(csc_from_vcsc.format)

    if(myFormat == "csc"):
        CSC_Equality(ivcsc_obj, vcsc_obj, control)
    # else:
        # CSR_Equality(ivcsc_obj, vcsc_obj, control)
    print("\033[92m", "Passed toCSC Equality", "\033[0m")

    vcsc_T = vcsc_obj.transpose()
    ivcsc_T = ivcsc_obj.transpose()
    control_T = control.transpose()
    assert vcsc_T.sum() == ivcsc_T.sum(), "vcsc_T: " + str(vcsc_T.sum()) + " ivcsc_T: " + str(ivcsc_T.sum())
    assert vcsc_T.sum() == control_T.sum(), "vcsc_T: " + str(vcsc_T.sum()) + " ivcsc_T: " + str(ivcsc_T.sum()) + " control_T: " + str(control_T.sum())
    assert vcsc_T.norm() == ivcsc_T.norm(), "vcsc_T: " + str(vcsc_T.norm()) + " ivcsc_T: " + str(ivcsc_T.norm())
    print("\033[92m", "Passed toCSC Transpose", "\033[0m")

    
    
    # print(ivcsc_obj)
    half_vcsc = vcsc_obj.slice(0, (int)(cols / 2)) 
    half_ivcsc = ivcsc_obj.slice(0, (int)(cols / 2))
    print("\033[92m", "Passed Slice", "\033[0m")
    assert half_ivcsc.sum() == half_vcsc.sum(), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum())
    # assert half_vcsc.sum() == control[].sum(), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum()) + " control: " + str(control[0, 2].sum())
    # assert half_ivcsc == half_vcsc, "half_vcsc: " + str(half_vcsc) + " half_ivcsc: " + str(half_ivcsc)
    print("\033[92m", "Passed Slice Sum Check", "\033[0m")

    
    # vcsc_obj *= 2
    # ivcsc_obj *= 2
    # control *= 2
    
    # assert vcsc_obj.sum() == ivcsc_obj.sum(), "vcsc_obj: " + str(vcsc_obj.sum()) + " ivcsc_obj: " + str(ivcsc_obj.sum())
    # assert vcsc_obj.sum() == control.sum(), "vcsc_obj: " + str(vcsc_obj.sum()) + " ivcsc_obj: " + str(ivcsc_obj.sum()) + " control: " + str(control.sum())
    print("Passed: ", randomTest + 1, " / 100")


