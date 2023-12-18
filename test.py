
import random

from matplotlib.pylab import f
from netaddr import P
import ivcsc
import vcsc
import scipy as sp
import numpy as np
import pytest

#TODO CSR doesn't work for toCSC() -> IVSparse needs to CSR
#TODO Make this do real unit testing
#TODO work on commented out tests
#TODO implement COO constructor testing
types = ( np.int32, np.uint32, np.int64, np.uint64) ## (np.int8, np.uint8, np.int16, np.uint16, , np.float32, np.float64)

indexTypes = (np.uint8, np.uint16, np.uint32, np.uint64)
# formats = ("csc", "csr")
# formats = ("csc",)
formats = ("csr",)
densities = (0.3, 0.4, 1.0)
rows = (1, 2, 10, 100)
cols = (1, 2, 10, 100)
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


    @pytest.fixture
    def csr_from_vcsc(self, VCSCMatrix):
        if(VCSCMatrix.major == "col"):
            pytest.skip("Skipping toCSR test for csc matrix")
        return VCSCMatrix.toCSR()

    @pytest.fixture
    def csr_from_ivcsc(self, IVCSCMatrix):   
        if(IVCSCMatrix.major == "col"):
            pytest.skip("Skipping toCSR test for csc matrix")
        return IVCSCMatrix.toCSR()

    @pytest.fixture
    def csc_from_vcsc(self, VCSCMatrix):
        if(VCSCMatrix.major == "row"):
            pytest.skip("Skipping toCSC test for csr matrix")
        return VCSCMatrix.toCSC()

    @pytest.fixture
    def csc_from_ivcsc(self, IVCSCMatrix):
        if(IVCSCMatrix.major == "row"):
            pytest.skip("Skipping toCSC test for csr matrix")
        return IVCSCMatrix.toCSC()


    
        
    # def testVectorLength(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     for x in range(SPMatrix.shape(1)):
    #         assert epsilon > abs(VCSCMatrix.vectorLength(x) - IVCSCMatrix.vectorLength(x)), "VCSCMatrix: " + str(VCSCMatrix.vectorLength(x)) + " IVCSCMatrix: " + str(IVCSCMatrix.vectorLength(x))
    #         # assert VCSCMatrix.vectorLength(x) == SPMatrix.getcol(x).sum(), "VCSCMatrix: " + str(VCSCMatrix.vectorLength(x)) + " IVCSCMatrix: " + str(IVCSCMatrix.vectorLength(x)) + " SPMatrix: " + str(SPMatrix.getrow(x).sum())

    def testSlice(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        if SPMatrix.shape[1] / 2 == 0:
            pytest.skip("Skipping slice test for would be 0 col matrix")

        half_vcsc = VCSCMatrix.slice(0, (int)(SPMatrix.shape[1] / 2)) 
        half_ivcsc = IVCSCMatrix.slice(0, (int)(SPMatrix.shape[1] / 2))
        assert epsilon > abs(half_ivcsc.sum() - half_vcsc.sum()), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum()) + " Diff: " + str(abs(half_ivcsc.sum() - half_vcsc.sum()))
        assert half_vcsc.shape() == half_ivcsc.shape(), "half_vcsc: " + str(half_vcsc.shape()) + " half_ivcsc: " + str(half_ivcsc.shape())
        # assert half_vcsc.sum() - SPMatrix[].sum(), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum()) + " SPMatrix: " + str(SPMatrix[0, 2].sum())
        
        half_sp = SPMatrix[:, 0:(int)(SPMatrix.shape[1] / 2)]
        assert epsilon > abs(half_sp.sum() - half_vcsc.sum()), "half_sp: " + str(half_sp.sum()) + " half_vcsc: " + str(half_vcsc.sum()) + " Diff: " + str(abs(half_sp.sum() - half_vcsc.sum()))
        assert half_sp.shape == half_vcsc.shape(), "half_sp: " + str(half_sp.shape) + " half_vcsc: " + str(half_vcsc.shape())

        result = half_vcsc.tocsc() - half_sp
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                assert epsilon > abs(result[x, y]), "half_vcsc: " + str(half_vcsc[x, y]) + " half_sp: " + str(half_sp[x, y]) + " Diff: " + str(abs(result[x, y]))
        

# if __name__ == "__main__":
    # test = tests()
    # test.testCSCConstructionIVCSC(SPMatrix)
    # test.testCSCConstructionVCSC(SPMatrix)
    # test.testVCSC_IVCSC_Equality(SPMatrix)
    # test.testSum(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testCopy(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testTrace(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testNorm(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testVectorLength(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testToCSC(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testTranspose(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testSlice(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testIPScalarMultiplyVCSC(SPMatrix, VCSCMatrix)
    # test.testScalarMultiplyVCSC(SPMatrix, VCSCMatrix)
    # test.testVectorMultiplyVCSC(SPVector, VCSCMatrix)
    # test.testMatrixMultiplyVCSC(SPMatrix, VCSCMatrix)
    # test.testIPScalarMultiplyIVCSC(SPMatrix, IVCSCMatrix)
    # test.testScalarMultiplyIVCSC(SPMatrix, IVCSCMatrix)
    # test.testVectorMultiplyIVCSC(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testMatrixMultiplyIVCSC(SPMatrix, IVCSCMatrix)
    # test.testToCSR(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testMajor(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testShape(SPMatrix, VCSCMatrix, IVCSCMatrix)
    # test.testDtype(SPMatrix, VCSCMatrix, IVCSCMatrix)
