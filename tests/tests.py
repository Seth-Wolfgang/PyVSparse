
import random

from matplotlib.pylab import f
import PyVSparse.ivcsc as ivcsc
import PyVSparse.vcsc as vcsc
import scipy as sp
import numpy as np
import pytest

#TODO CSR doesn't work for toCSC() -> IVSparse needs to CSR
#TODO Make this do real unit testing
#TODO work on commented out tests
#TODO implement COO constructor testing
# types = ( np.int32, np.uint32, np.int64, np.uint64, np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64) ## ()
types = (np.int32,)
indexTypes = (np.uint8, np.uint16, np.uint32, np.uint64)
formats = ("csc", "csr")
# formats = ("csc",)
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
        return vcsc.VCSC(SPMatrix)#, indexT = request.param)

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

    def testDtype(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        assert VCSCMatrix.dtype == SPMatrix.dtype, "VCSCMatrix: " + str(VCSCMatrix.dtype) + " SPMatrix: " + str(SPMatrix.dtype)
        assert IVCSCMatrix.dtype == SPMatrix.dtype, "IVCSCMatrix: " + str(IVCSCMatrix.dtype) + " SPMatrix: " + str(SPMatrix.dtype)

    def testShape(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        assert VCSCMatrix.shape() == SPMatrix.shape, "VCSCMatrix: " + str(VCSCMatrix.shape) + " SPMatrix: " + str(SPMatrix.shape)
        assert IVCSCMatrix.shape() == SPMatrix.shape, "IVCSCMatrix: " + str(IVCSCMatrix.shape) + " SPMatrix: " + str(SPMatrix.shape)

    def testMajor(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        if SPMatrix.format == "CSC":
            assert VCSCMatrix.major == "Col", "VCSCMatrix: " + str(VCSCMatrix.major) + " myFormat: " + str(formats)
            assert IVCSCMatrix.major == "Col", "IVCSCMatrix: " + str(IVCSCMatrix.major) + " myFormat: " + str(formats)
        elif SPMatrix.format == "CSR":
            assert VCSCMatrix.major == "Row", "VCSCMatrix: " + str(VCSCMatrix.major) + " myFormat: " + str(formats)
            assert IVCSCMatrix.major == "Row", "IVCSCMatrix: " + str(IVCSCMatrix.major) + " myFormat: " + str(formats)

    def testToCSR(self, IVCSCMatrix, VCSCMatrix, SPMatrix):
        csc_from_ivcsc = IVCSCMatrix.tocsr()
        csc_from_vcsc = VCSCMatrix.tocsr()

        resultvcsc = csc_from_vcsc - SPMatrix
        resultivcsc = csc_from_ivcsc - SPMatrix

        assert epsilon > abs(resultvcsc.sum() - resultivcsc.sum()), "resultvcsc: " + str(resultvcsc.sum()) + " resultivcsc: " + str(resultivcsc.sum())
        assert epsilon > abs(csc_from_ivcsc.sum() - csc_from_vcsc.sum()), "csc_from_ivcsc: " + str(csc_from_ivcsc.sum()) + " csc_from_vcsc: " + str(csc_from_vcsc.sum())
        
        for x in range(csc_from_ivcsc.shape[0]):
            for y in range(csc_from_ivcsc.shape[1]):
                assert epsilon > abs(csc_from_ivcsc[x, y] - csc_from_vcsc[x, y]), "csc_from_ivcsc: " + str(csc_from_ivcsc[x, y]) + " csc_from_vcsc: " + str(csc_from_vcsc[x, y]) + " x: " + str(x) + " y: " + str(y)

    def testToCSC(self, IVCSCMatrix, VCSCMatrix, SPMatrix):
        csc_from_ivcsc = IVCSCMatrix.tocsc()
        csc_from_vcsc = VCSCMatrix.tocsc()

        resultvcsc = csc_from_vcsc - SPMatrix
        resultivcsc = csc_from_ivcsc - SPMatrix

        assert epsilon > abs(resultvcsc.sum() - resultivcsc.sum()), "resultvcsc: " + str(resultvcsc.sum()) + " resultivcsc: " + str(resultivcsc.sum())
        assert epsilon > abs(csc_from_ivcsc.sum() - csc_from_vcsc.sum()), "csc_from_ivcsc: " + str(csc_from_ivcsc.sum()) + " csc_from_vcsc: " + str(csc_from_vcsc.sum())
        
        for x in range(csc_from_ivcsc.shape[0]):
            for y in range(csc_from_ivcsc.shape[1]):
                assert epsilon > abs(csc_from_ivcsc[x, y] - csc_from_vcsc[x, y]), "csc_from_ivcsc: " + str(csc_from_ivcsc[x, y]) + " csc_from_vcsc: " + str(csc_from_vcsc[x, y]) + " x: " + str(x) + " y: " + str(y)


    def testCSCConstructionVCSC(self, SPMatrix):
        test = vcsc.VCSC(SPMatrix)
        assert epsilon > abs(test.sum() - SPMatrix.sum()), "test: " + str(test.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    def testCSCConstructionIVCSC(self, SPMatrix):
        test = ivcsc.IVCSC(SPMatrix)
        assert epsilon > abs(test.sum() - SPMatrix.sum()), "test: " + str(test.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    def testVCSC_IVCSC_Equality(self, SPMatrix):
        VCSCMatrix = vcsc.VCSC(SPMatrix)
        IVCSCMatrix = ivcsc.IVCSC(SPMatrix)
        assert epsilon > abs(VCSCMatrix.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
        assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    def testSum(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        assert epsilon > abs(SPMatrix.sum() - VCSCMatrix.sum())
        assert epsilon > abs(SPMatrix.sum() - IVCSCMatrix.sum())


    def testCopy(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        VCSCMatrix_copy = VCSCMatrix
        IVCSCMatrix_copy = IVCSCMatrix

        assert VCSCMatrix_copy == VCSCMatrix
        assert IVCSCMatrix_copy == IVCSCMatrix

        assert epsilon > abs(VCSCMatrix_copy.sum() - VCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
        assert epsilon > abs(IVCSCMatrix_copy.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())

    def testNorm(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        val = sp.sparse.linalg.norm(SPMatrix, "fro")
        assert epsilon > abs(VCSCMatrix.norm() - IVCSCMatrix.norm())
        assert epsilon > abs(VCSCMatrix.norm() - val), "VCSCMatrix: " + str(VCSCMatrix.norm()) + " IVCSCMatrix: " + str(IVCSCMatrix.norm()) + " SPMatrix: " + str(val)

    def testTranspose(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        vcsc_T = VCSCMatrix.transpose()
        ivcsc_T = IVCSCMatrix.transpose()
        SPMatrix_T = SPMatrix.transpose()
        assert epsilon > abs(vcsc_T.sum() - ivcsc_T.sum()), "vcsc_T: " + str(vcsc_T.sum()) + " ivcsc_T: " + str(ivcsc_T.sum())
        assert epsilon > abs(vcsc_T.sum() - SPMatrix_T.sum()), "vcsc_T: " + str(vcsc_T.sum()) + " ivcsc_T: " + str(ivcsc_T.sum()) + " SPMatrix_T: " + str(SPMatrix_T.sum())
        assert epsilon > abs(vcsc_T.norm() - ivcsc_T.norm()), "vcsc_T: " + str(vcsc_T.norm()) + " ivcsc_T: " + str(ivcsc_T.norm())

    def testInPlaceTranspose(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        VCSCMatrix.transpose(inplace = True)
        IVCSCMatrix.transpose(inplace = True)
        SPMatrix.transpose()
        assert epsilon > abs(VCSCMatrix.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
        assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())
        assert epsilon > abs(VCSCMatrix.norm() - IVCSCMatrix.norm()), "VCSCMatrix: " + str(VCSCMatrix.norm()) + " IVCSCMatrix: " + str(IVCSCMatrix.norm())

    def testIPScalarMultiplyVCSC(self, SPMatrix, VCSCMatrix):
        VCSCMatrix *= 2
        SPMatrix *= 2

        assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(VCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())

    def testScalarMultiplyVCSC(self, SPMatrix, VCSCMatrix):
        VCSCresult = VCSCMatrix * 2
        SPresult = SPMatrix * 2

        assert epsilon > abs(VCSCresult.sum() - SPresult.sum())
        assert VCSCresult.shape() == SPresult.shape

    def testIPScalarMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCMatrix *= 2
        SPMatrix *= 2

        assert epsilon > abs(IVCSCMatrix.sum() - SPMatrix.sum()), " IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())
        assert IVCSCMatrix.shape() == SPMatrix.shape, " IVCSCMatrix: " + str(IVCSCMatrix.shape) + " SPMatrix: " + str(SPMatrix.shape)

    def testScalarMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCresult = IVCSCMatrix * 2
        SPresult = SPMatrix * 2

        assert epsilon > abs(IVCSCresult.sum() - SPresult.sum()), "IVCSCresult: " + str(IVCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
        assert IVCSCresult.shape() == SPresult.shape, "IVCSCresult: " + str(IVCSCresult.shape) + " SPresult: " + str(SPresult.shape)

    def testMatrixMultiplyVCSC(self, SPMatrix, VCSCMatrix):
        VCSCresult = VCSCMatrix *  SPMatrix.transpose().toarray()
        SPresult = SPMatrix *  SPMatrix.transpose().toarray()

        # assert epsilon > abs(VCSCresult.sum() - SPresult.sum()), "VCSCresult: " + str(VCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
        result = VCSCresult - SPresult # SHOULD be a matrix of all zeros
        for x in result:
            for y in x:
                assert epsilon > abs(y), "VCSCresult: " + str(VCSCresult) + " SPresult: " + str(SPresult)

    def testMatrixMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCresult = IVCSCMatrix * SPMatrix.transpose().toarray()
        SPresult = SPMatrix * SPMatrix.transpose().toarray()

        print("Shape", IVCSCMatrix.shape())

        # assert epsilon > abs(IVCSCresult.sum() - SPresult.sum()), "IVCSCresult: " + str(IVCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
        result = IVCSCresult - SPresult # SHOULD be a matrix of all zeros
        for x in result:
            for y in x:
                assert epsilon > abs(y), "IVCSCresult: " + str(IVCSCresult) + " SPresult: " + str(SPresult)


    def testAppendOwnFormat(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        VCSCMat2 = vcsc.VCSC(SPMatrix)
        IVCSCMat2 = ivcsc.IVCSC(SPMatrix)   
        VCSCMat2.append(VCSCMatrix)
        IVCSCMat2.append(IVCSCMatrix)        

        VCSC2CSC = VCSCMat2.tocsc()
        IVCSC2CSC = IVCSCMat2.tocsc()

        result = (VCSC2CSC - IVCSC2CSC).todense()
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                assert epsilon > abs(result[x, y]), "VCSC2CSC: " + str(VCSC2CSC[x, y]) + " IVCSC2CSC: " + str(IVCSC2CSC[x, y]) + " Diff: " + str(abs(result[x, y]))
