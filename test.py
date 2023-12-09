
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
types = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32)
# (np.int64, np.uint64, np.float32, np.float64)
indexTypes = (np.uint8, np.uint16, np.uint32, np.uint64)
# formats = ("csc", "csr")
formats = ("csc",)
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
        return vcsc.VCSC(SPMatrix, indexT = request.param)

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

    # def testDtype(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     assert VCSCMatrix.dtype == SPMatrix.dtype, "VCSCMatrix: " + str(VCSCMatrix.dtype) + " SPMatrix: " + str(SPMatrix.dtype)
    #     assert IVCSCMatrix.dtype == SPMatrix.dtype, "IVCSCMatrix: " + str(IVCSCMatrix.dtype) + " SPMatrix: " + str(SPMatrix.dtype)

    # def testShape(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     assert VCSCMatrix.shape == SPMatrix.shape, "VCSCMatrix: " + str(VCSCMatrix.shape) + " SPMatrix: " + str(SPMatrix.shape)
    #     assert IVCSCMatrix.shape == SPMatrix.shape, "IVCSCMatrix: " + str(IVCSCMatrix.shape) + " SPMatrix: " + str(SPMatrix.shape)

    # def testMajor(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     if SPMatrix.format == "CSC":
    #         assert VCSCMatrix.major == "Col", "VCSCMatrix: " + str(VCSCMatrix.major) + " myFormat: " + str(formats)
    #         assert IVCSCMatrix.major == "Col", "IVCSCMatrix: " + str(IVCSCMatrix.major) + " myFormat: " + str(formats)
    #     elif SPMatrix.format == "CSR":
    #         assert VCSCMatrix.major == "Row", "VCSCMatrix: " + str(VCSCMatrix.major) + " myFormat: " + str(formats)
    #         assert IVCSCMatrix.major == "Row", "IVCSCMatrix: " + str(IVCSCMatrix.major) + " myFormat: " + str(formats)


    # def testCSCConstructionVCSC(self, SPMatrix):
    #     test = vcsc.VCSC(SPMatrix)
    #     assert epsilon > abs(test.sum() - SPMatrix.sum()), "test: " + str(test.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    # def testCSCConstructionIVCSC(self, SPMatrix):
    #     test = ivcsc.IVCSC(SPMatrix)
    #     assert epsilon > abs(test.sum() - SPMatrix.sum()), "test: " + str(test.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    # def testVCSC_IVCSC_Equality(self, SPMatrix):
    #     VCSCMatrix = vcsc.VCSC(SPMatrix)
    #     IVCSCMatrix = ivcsc.IVCSC(SPMatrix)
    #     assert epsilon > abs(VCSCMatrix.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
    #     assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    # def CSR_Equality(self, ivcsc, vcsc, SPMatrix):
    #     csc_from_ivcsc = ivcsc.toCSR
    #     csc_from_vcsc = vcsc.toCSR

    #     for x, y, z in zip(csc_from_ivcsc.indices, csc_from_vcsc.indices, SPMatrix.indices):
    #         assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y)
    #         assert x == z, "csc_from_ivcsc: " + str(x) + " SPMatrix: " + str(z)

    #     for x, y, z in zip(csc_from_ivcsc.indptr, csc_from_vcsc.indptr, SPMatrix.indptr):
    #         assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " SPMatrix: " + str(z)
    #         assert x == z, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " SPMatrix: " + str(z)
    #     for x, y, z in zip(csc_from_ivcsc.data, csc_from_vcsc.data, SPMatrix.data):
    #         assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " SPMatrix: " + str(z)
    #         assert x == z, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y) + " SPMatrix: " + str(z)

    # def testSum(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     assert epsilon > abs(SPMatrix.sum() - VCSCMatrix.sum())
    #     assert epsilon > abs(SPMatrix.sum() - IVCSCMatrix.sum())


    # def testCopy(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     VCSCMatrix_copy = VCSCMatrix
    #     IVCSCMatrix_copy = IVCSCMatrix

    #     assert VCSCMatrix_copy == VCSCMatrix
    #     assert IVCSCMatrix_copy == IVCSCMatrix

    #     assert epsilon > abs(VCSCMatrix_copy.sum() - VCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
    #     assert epsilon > abs(IVCSCMatrix_copy.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())

    # def testTrace(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     if rows != cols:c
    #         pytest.skip("Skipping trace test for non-square matrix")
    #     assert VCSCMatrix.trace() == IVCSCMatrix.trace(), "VCSCMatrix: " + str(VCSCMatrix.trace()) + " IVCSCMatrix: " + str(IVCSCMatrix.trace())
    #     assert VCSCMatrix.trace() == SPMatrix.trace(), "VCSCMatrix: " + str(VCSCMatrix.trace()) + " IVCSCMatrix: " + str(IVCSCMatrix.trace()) + " SPMatrix: " + str(SPMatrix.trace())

    # def testNorm(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     assert epsilon > abs(VCSCMatrix.norm() - IVCSCMatrix.norm())

    # def testVectorLength(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     for x in range(SPMatrix.shape(1)):
    #         assert epsilon > abs(VCSCMatrix.vectorLength(x) - IVCSCMatrix.vectorLength(x)), "VCSCMatrix: " + str(VCSCMatrix.vectorLength(x)) + " IVCSCMatrix: " + str(IVCSCMatrix.vectorLength(x))
    #         # assert VCSCMatrix.vectorLength(x) == SPMatrix.getcol(x).sum(), "VCSCMatrix: " + str(VCSCMatrix.vectorLength(x)) + " IVCSCMatrix: " + str(IVCSCMatrix.vectorLength(x)) + " SPMatrix: " + str(SPMatrix.getrow(x).sum())

    # def testToCSC(self, SPMatrix, csc_from_ivcsc, csc_from_vcsc):
    #     if(SPMatrix.format == "csr"):
    #         pytest.skip("Skipping toCSC test for csr matrix")
    
    #     assert epsilon > abs(csc_from_ivcsc.sum() - csc_from_vcsc.sum()), "csc_from_ivcsc: " + str(IVCSCMatrix.sum()) + " csc_from_vcsc: " + str(VCSCMatrix.sum())
    #     assert csc_from_ivcsc.shape == csc_from_vcsc.shape, "csc_from_ivcsc: " + str(csc_from_ivcsc.shape) + " csc_from_vcsc: " + str(csc_from_vcsc.shape)
    #     assert csc_from_ivcsc.dtype == csc_from_vcsc.dtype, "csc_from_ivcsc: " + str(csc_from_ivcsc.dtype) + " csc_from_vcsc: " + str(csc_from_vcsc.dtype)
    #     assert csc_from_ivcsc.format == csc_from_vcsc.format, "csc_from_ivcsc: " + str(csc_from_ivcsc.format) + " csc_from_vcsc: " + str(csc_from_vcsc.format)

    #     x = csc_from_ivcsc.__str__()
    #     y = csc_from_vcsc.__str__()
    #     z = SPMatrix.__str__()
    #     assert x == y, "csc_from_ivcsc: " + str(x) + " csc_from_vcsc: " + str(y)
    #     assert x == z, "csc_from_ivcsc: " + str(x) + " SPMatrix: " + str(z)

    # def testToCSR(self, SPMatrix, csr_from_ivcsc, csr_from_vcsc):
    #     if(SPMatrix.format == "csc"):
    #         pytest.skip("Skipping toCSR test for csc matrix")

    #     assert epsilon > abs(csr_from_ivcsc.sum() - csr_from_vcsc.sum()), "csr_from_ivcsc: " + str(csr_from_ivcsc.sum()) + " csr_from_vcsc: " + str(csr_from_vcsc.sum())
    #     assert csr_from_ivcsc.shape == csr_from_vcsc.shape, "csr_from_ivcsc: " + str(csr_from_ivcsc.shape) + " csr_from_vcsc: " + str(csr_from_vcsc.shape)
    #     assert csr_from_ivcsc.dtype == csr_from_vcsc.dtype, "csr_from_ivcsc: " + str(csr_from_ivcsc.dtype) + " csr_from_vcsc: " + str(csr_from_vcsc.dtype)
    #     assert csr_from_ivcsc.format == csr_from_vcsc.format, "csr_from_ivcsc: " + str(csr_from_ivcsc.format) + " csr_from_vcsc: " + str(csr_from_vcsc.format)

    #     x = csr_from_ivcsc.__str__()
    #     y = csr_from_vcsc.__str__()
    #     z = SPMatrix.__str__()
    #     assert x == y, "csr_from_ivcsc: " + str(x) + " csr_from_vcsc: " + str(y)
    #     assert x == z, "csr_from_ivcsc: " + str(x) + " SPMatrix: " + str(z)

    # def testTranspose(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     vcsc_T = VCSCMatrix.transpose()
    #     ivcsc_T = IVCSCMatrix.transpose()
    #     SPMatrix_T = SPMatrix.transpose()
    #     assert epsilon > abs(vcsc_T.sum() - ivcsc_T.sum()), "vcsc_T: " + str(vcsc_T.sum()) + " ivcsc_T: " + str(ivcsc_T.sum())
    #     assert epsilon > abs(vcsc_T.sum() - SPMatrix_T.sum()), "vcsc_T: " + str(vcsc_T.sum()) + " ivcsc_T: " + str(ivcsc_T.sum()) + " SPMatrix_T: " + str(SPMatrix_T.sum())
    #     assert epsilon > abs(vcsc_T.norm() - ivcsc_T.norm()), "vcsc_T: " + str(vcsc_T.norm()) + " ivcsc_T: " + str(ivcsc_T.norm())

    # def testInPlaceTranspose(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
    #     print(SPMatrix.shape, VCSCMatrix.shape, IVCSCMatrix.shape)
    #     print(SPMatrix.sum(), VCSCMatrix.sum(), IVCSCMatrix.sum())
    #     print(SPMatrix.nnz, VCSCMatrix.nnz, IVCSCMatrix.nnz)
    #     print(SPMatrix.dtype, VCSCMatrix.dtype, IVCSCMatrix.dtype)

    #     VCSCMatrix.transpose(inplace = True)
    #     IVCSCMatrix.transpose(inplace = True)
    #     SPMatrix.transpose(inplace = True)
    #     assert epsilon > abs(VCSCMatrix.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
    #     assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())
    #     assert epsilon > abs(VCSCMatrix.norm() - IVCSCMatrix.norm()), "VCSCMatrix: " + str(VCSCMatrix.norm()) + " IVCSCMatrix: " + str(IVCSCMatrix.norm())

    # def testSlice(self, SPMatrix, VCSCMatrix, IVCSCMatrix):    
    #     half_vcsc = VCSCMatrix.slice(0, (int)(SPMatrix.shape[1] / 2)) 
    #     half_ivcsc = IVCSCMatrix.slice(0, (int)(SPMatrix.shape[1] / 2))
    #     assert epsilon > abs(half_ivcsc.sum() - half_vcsc.sum()), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum()) + " Diff: " + str(abs(half_ivcsc.sum() - half_vcsc.sum()))
    #     # assert half_vcsc.sum() - SPMatrix[].sum(), "half_vcsc: " + str(half_vcsc.sum()) + " half_ivcsc: " + str(half_ivcsc.sum()) + " SPMatrix: " + str(SPMatrix[0, 2].sum())
        
    #     x = half_vcsc.__str__()
    #     y = half_ivcsc.__str__()

        # for i in range(half_vcsc.shape[0]):
            # for j in range(half_vcsc.shape[1]):
                # assert epsilon > abs(half_vcsc[i, j] - half_ivcsc[i, j]), "half_vcsc: " + str(half_vcsc[i, j]) + " half_ivcsc: " + str(half_ivcsc[i, j]) + " i: " + str(i) + " j: " + str(j)

        # assert x == y, "half_vcsc: " + str(x) + " half_ivcsc: " + str(y)
        # assert half_ivcsc == half_vcsc, "half_vcsc: " + str(half_vcsc) + " half_ivcsc: " + str(half_ivcsc)

    # def testIPScalarMultiplyVCSC(self, SPMatrix, VCSCMatrix):
    #     VCSCMatrix *= 2
    #     SPMatrix *= 2
    #     print(VCSCMatrix.sum())
    #     print(SPMatrix.sum())

    #     assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(VCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    # def testScalarMultiplyVCSC(self, SPMatrix, VCSCMatrix):
    #     VCSCresult = VCSCMatrix * 2
    #     SPresult = SPMatrix * 2

    #     assert epsilon > abs(VCSCresult.sum() - SPresult.sum())
    #     assert VCSCresult.shape == SPresult.shape

    # def testVectorMultiplyVCSC(self, SPVector, VCSCMatrix, SPMatrix):
    #     VCSCresult = VCSCMatrix *  SPVector
    #     SPresult = SPMatrix *  SPVector

    #     assert epsilon > abs(VCSCresult.sum() - SPresult.sum()), "VCSCresult: " + str(VCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
    #     assert VCSCresult.shape == SPresult.shape, "VCSCresult: " + str(VCSCresult.shape) + " SPresult: " + str(SPresult.shape)

    # def testMatrixMultiplyVCSC(self, SPMatrix, VCSCMatrix):
    #     VCSCresult = VCSCMatrix *  SPMatrix.transpose().toarray()
    #     SPresult = SPMatrix *  SPMatrix.transpose().toarray()

    #     assert epsilon > abs(VCSCresult.sum() - SPresult.sum()), "VCSCresult: " + str(VCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
    #     assert VCSCresult.shape == SPresult.shape, "VCSCresult: " + str(VCSCresult.shape) + " SPresult: " + str(SPresult.shape)

    # def testIPScalarMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
    #     IVCSCMatrix *= 2
    #     SPMatrix *= 2

    #     assert epsilon > abs(IVCSCMatrix.sum() - SPMatrix.sum()), " IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())
    #     assert all(IVCSCMatrix.shape == SPMatrix.shape), " IVCSCMatrix: " + str(IVCSCMatrix.shape) + " SPMatrix: " + str(SPMatrix.shape)

    # def testScalarMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
    #     IVCSCresult = IVCSCMatrix * 2
    #     SPresult = SPMatrix * 2

    #     assert epsilon > abs(IVCSCresult.sum() - SPresult.sum()), "IVCSCresult: " + str(IVCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
    #     assert IVCSCresult.shape == SPresult.shape, "IVCSCresult: " + str(IVCSCresult.shape) + " SPresult: " + str(SPresult.shape)

    # def testVectorMultiplyIVCSC(self, SPMatrix, IVCSCMatrix, SPVector):
    #     IVCSCresult = IVCSCMatrix * SPVector
    #     SPresult = SPMatrix * SPVector

    #     assert epsilon > abs(IVCSCresult.sum() - SPresult.sum()), "IVCSCresult: " + str(IVCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
    #     assert IVCSCresult.shape == SPresult.shape, "IVCSCresult: " + str(IVCSCresult.shape) + " SPresult: " + str(SPresult.shape)

    def testMatrixMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCresult = IVCSCMatrix * SPMatrix.transpose().toarray()
        SPresult = SPMatrix * SPMatrix.transpose().toarray()
# 
        assert epsilon > abs(IVCSCresult.sum() - SPresult.sum()), "IVCSCresult: " + str(IVCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
        assert IVCSCresult.shape == SPresult.shape, "IVCSCresult: " + str(IVCSCresult.shape) + " SPresult: " + str(SPresult.shape)


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
