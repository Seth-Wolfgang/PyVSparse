
import random

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
types = ( np.int32,)# np.uint32, np.int64, np.uint64, np.int8, np.uint8, np.int16, np.uint16 , np.float32, np.float64)

indexTypes = (np.uint8, np.uint16, np.uint32, np.uint64)
# formats = ("csc", "csr")
formats = ("csc",)
# formats = ("csr",)
# densities = (0.3, 0.4, 1.0)
densities = (1.0,)
# rows = (1, 2, 10, 100)
rows = (2, 100)
# cols = (1, 2, 10, 100)
cols = (2, 100)
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
        return vcsc.VCSC(SPMatrix)

    @pytest.fixture
    def IVCSCMatrix(self, SPMatrix):
        return ivcsc.IVCSC(SPMatrix)

    @pytest.fixture
    # @pytest.mark.parametrize('densities', densities)
    def SPVector(self, SPMatrix):
        return np.ones((SPMatrix.shape[1], 1))  

 


    def testSlice(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        if SPMatrix.shape[1] / 2 == 0:
            pytest.skip("Skipping slice test for would be 0 col matrix")

        half_vcsc = VCSCMatrix.slice(0, (int)(SPMatrix.shape[1] / 2)) 
        half_ivcsc = IVCSCMatrix.slice(0, (int)(SPMatrix.shape[1] / 2))
        assert epsilon > abs(half_ivcsc.sum() - half_vcsc.sum()), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum()) + " Diff: " + str(abs(half_ivcsc.sum() - half_vcsc.sum()))
        assert half_vcsc.shape() == half_ivcsc.shape(), "half_vcsc: " + str(half_vcsc.shape()) + " half_ivcsc: " + str(half_ivcsc.shape())
        
        half_sp = SPMatrix[:, 0:(int)(SPMatrix.shape[1] / 2)]
        assert epsilon > abs(half_sp.sum() - half_vcsc.sum()), "half_sp: " + str(half_sp.sum()) + " half_vcsc: " + str(half_vcsc.sum()) + " Diff: " + str(abs(half_sp.sum() - half_vcsc.sum()))
        assert half_sp.shape == half_vcsc.shape(), "half_sp: " + str(half_sp.shape) + " half_vcsc: " + str(half_vcsc.shape())

        result = (half_vcsc.tocsc() - half_sp).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((half_sp.shape[0], half_sp.shape[1])), decimal=3, verbose=True)
