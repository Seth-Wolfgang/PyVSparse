
from curses.ascii import SP
import os
import random
import sys
from unittest import result

from matplotlib.pylab import f
from netaddr import P
import PyVSparse.ivcsc as ivcsc
import PyVSparse.vcsc as vcsc
import scipy as sp
import numpy as np
import pytest

#TODO CSR doesn't work for toCSC() -> IVSparse needs to CSR
#TODO Make this do real unit testing
#TODO work on commented out tests
#TODO implement COO constructor testing
types = ( np.int32, np.uint32, np.int64, np.uint64, np.int8, np.uint8, np.int16, np.uint16 , np.float32, np.float64)

indexTypes = (np.uint8, np.uint16, np.uint32, np.uint64)
formats = ("csc", "csr")
# formats = ("csc",)
# formats = ("csr",)
# densities = (0.3, 0.4, 1.0)
densities = (1.0,)
# rows = (1, 2, 10, 100)
rows = (5, 30, 100)
# cols = (1, 2, 10, 100)
cols = (5, 15, 100)
epsilon = 1e-3

cases = []
for type in types:
    for density in densities:
        for format in formats:
            for row in rows:
                for col in cols:
                    cases.append((type, density, format, row, col))

class Test:

    @pytest.fixture(params=cases)
    def SPMatrix(self, request):
        myType, densities, formats, rows, cols = request.param
        self.format = formats
        nnz = int(rows * cols * densities + 1)

        if myType == np.float32 or myType == np.float64:
            mat = [[0.0 for x in range(cols)] for y in range(rows)]
            for x in range(nnz):
                mat[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = random.random()
        else:
            mat = [[0 for x in range(cols)] for y in range(rows)]
            for x in range(nnz):
                mat[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = random.randint(0, 100)

        if formats == "csc":
            mock = sp.sparse.csc_matrix(mat, dtype = myType)
        else:
            mock = sp.sparse.csr_matrix(mat, dtype = myType)
        if mock.nnz == 0:
            mock[0, 0] = 1
        return mock
    
    @pytest.fixture(params=indexTypes)
    def VCSCMatrix(self, SPMatrix, request):
        # print(request.param)
        return vcsc.VCSC(SPMatrix, indexType=request.param)

    @pytest.fixture
    def IVCSCMatrix(self, SPMatrix):
        return ivcsc.IVCSC(SPMatrix)

    @pytest.fixture
    # @pytest.mark.parametrize('densities', densities)
    def SPVector(self, SPMatrix):
        return np.ones((SPMatrix.shape[1], 1))  


    # def testRandomAccessVCSC(self, SPMatrix, VCSCMatrix):

    #     for x in range(SPMatrix.shape[0]):
    #         for y in range(SPMatrix.shape[1]):
    #             assert VCSCMatrix[x, y] == SPMatrix[x, y]
        

 


    # def testSlice(self, SPMatrix, VCSCMatrix):
    #     if SPMatrix.shape[1] // 2 == 0:
    #         pytest.skip("Skipping slice test for would be 0 col matrix")

    #     half_vcsc = VCSCMatrix.slice(0, SPMatrix.shape[1] // 2)
    #     half_sp = SPMatrix[:, 0:(int)(SPMatrix.shape[1] // 2)]

    #     result = half_vcsc.tocsc()

    #     np.testing.assert_array_almost_equal(result.toarray(), half_sp.toarray(), decimal=3, verbose=True)

    # def testSliceIVCSC(self, SPMatrix, IVCSCMatrix):
    #     if SPMatrix.shape[1] / 2 == 0:
    #         pytest.skip("Skipping slice test for would be 0 col matrix")
 
    #     half_ivcsc = IVCSCMatrix.slice(0, SPMatrix.shape[1] // 2)
    #     half_sp = SPMatrix[:, 0: SPMatrix.shape[1] // 2]

    #     result = half_ivcsc.tocsc()

    #     np.testing.assert_array_almost_equal(result.toarray(), half_sp.toarray(), decimal=3, verbose=True)

   # def testMinColVCSC(self, SPMatrix, VCSCMatrix): 
    #     vcsc_min = VCSCMatrix.min(axis=0)

    #     # https://stackoverflow.com/a/49389908/12895299
    #     SPArray = SPMatrix.toarray()


    #     # np.testing.assert_almost_equal(vcsc_min, minval, decimal=3)

    # def testMinColIVCSC(self, SPMatrix, IVCSCMatrix):
    #     ivcsc_min = IVCSCMatrix.min(axis=0)

    #     # https://stackoverflow.com/a/49389908/12895299
    #     SPArray = SPMatrix.toarray()
    #     minval = np.min(np.where(SPArray==0, SPArray.max(), SPArray), axis=0)

    #     # turn it into a 2d array 
    #     minval = np.reshape(minval, (1, SPMatrix.shape[1]))

    #     # The code above removes zeros that SHOULD be there (zero vector in matrix)
    #     # This code adds them back in
    #     for x in range(len(ivcsc_min)):
    #         if ivcsc_min[0, x] == 0:
    #             minval[0, x] = 0

    #     result = ivcsc_min - minval

    #     assert np.allclose(result, 0, atol=epsilon)

        # np.testing.assert_almost_equal(ivcsc_min, minval, decimal=3)
    
    # def testMinRowVCSC(self, SPMatrix, VCSCMatrix):
    #     vcsc_min = VCSCMatrix.min(axis=1)
        
    #     # https://stackoverflow.com/a/49389908/12895299
    #     SPArray = SPMatrix.toarray()
    #     minval = np.min(np.where(SPArray==0, SPArray.max(), SPArray), axis=1)

    #     # turn it into a 2d array 
    #     minval = np.reshape(minval, (SPMatrix.shape[0], 1))

    #     # The code above removes zeros that SHOULD be there (zero vector in matrix)
    #     # This code adds them back in
    #     for x in range(len(vcsc_min)):
    #         if vcsc_min[x, 0] == 0:
    #             minval[x, 0] = 0

    #     result = vcsc_min - minval

    #     assert np.allclose(result, 0, atol=epsilon)

        # np.testing.assert_almost_equal(vcsc_min, minval, decimal=3)

    # def testMinRowIVCSC(self, SPMatrix, IVCSCMatrix):
    #     ivcsc_min = IVCSCMatrix.min(axis=1)
    #     # https://stackoverflow.com/a/49389908/12895299
    #     SPArray = SPMatrix.toarray()
    #     minval = np.min(np.where(SPArray==0, SPArray.max(), SPArray), axis=1)

    #     # turn it into a 2d array 
    #     minval = np.reshape(minval, (SPMatrix.shape[0], 1))

    #     # The code above removes zeros that SHOULD be there (zero vector in matrix)
    #     # This code adds them back in
    #     for x in range(len(ivcsc_min)):
    #         if ivcsc_min[x, 0] == 0:
    #             minval[x, 0] = 0

    #     result = ivcsc_min - minval

    #     assert np.allclose(result, 0, atol=epsilon)

    #     np.testing.assert_almost_equal(ivcsc_min, minval, decimal=3)

