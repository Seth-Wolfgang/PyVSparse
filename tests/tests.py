
import os
import random
from PyVSparse import VCSC

import PyVSparse.ivcsc as ivcsc
import PyVSparse.vcsc as vcsc
import scipy as sp
import numpy as np
import pytest

#TODO CSR doesn't work for tocsc() -> IVSparse needs to CSR
#TODO Make this do real unit testing
#TODO work on commented out tests
#TODO implement COO constructor testing
types = ( np.int32, np.int64, np.uint64, np.int8, np.int16, np.float32, np.float64) ## ( )
# types = (np.int32,)
indexTypes = (np.uint32, np.uint64, np.uint8, np.uint16, )
formats = ("csc", "csr")
# formats = ("csc",)
densities = (1.0,)
rows = (1, 100)
cols = (1, 100)
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

    def testtocscIVCSC(self, IVCSCMatrix, SPMatrix):
        csc_from_ivcsc = IVCSCMatrix.tocsc()

        result = csc_from_ivcsc - SPMatrix

        assert epsilon > abs(result.sum()), "resultivcsc: " + str(csc_from_ivcsc.sum())
        
        for x in range(SPMatrix.shape[0]):
            for y in range(SPMatrix.shape[1]):
                assert epsilon > abs(SPMatrix[x, y] - csc_from_ivcsc[x, y]), "SPMatrix: " + str(SPMatrix[x, y]) + " csc_from_ivcsc: " + str(csc_from_ivcsc[x, y]) + " x: " + str(x) + " y: " + str(y)

    def testtocsrIVCSC(self, IVCSCMatrix, SPMatrix):
        csr_from_ivcsc = IVCSCMatrix.tocsc()

        result = csr_from_ivcsc - SPMatrix

        assert epsilon > abs(result.sum()), "resultivcsc: " + str(csr_from_ivcsc.sum())
        
        for x in range(SPMatrix.shape[0]):
            for y in range(SPMatrix.shape[1]):
                assert epsilon > abs(SPMatrix[x, y] - csr_from_ivcsc[x, y]), "SPMatrix: " + str(SPMatrix[x, y]) + " csc_from_ivcsc: " + str(csr_from_ivcsc[x, y]) + " x: " + str(x) + " y: " + str(y)


    def testtocscVCSC(self, VCSCMatrix, SPMatrix):
        # print(VCSCMatrix)

        csc_from_vcsc = VCSCMatrix.tocsc()


        result = csc_from_vcsc - SPMatrix

        assert epsilon > abs(result.sum()), "resultvcsc: " + str(csc_from_vcsc.sum())
        
        for x in range(SPMatrix.shape[0]):
            for y in range(SPMatrix.shape[1]):
                assert epsilon > abs(SPMatrix[x, y] - csc_from_vcsc[x, y]), "SPMatrix: " + str(SPMatrix[x, y]) + " csc_from_vcsc: " + str(csc_from_vcsc[x, y]) + " x: " + str(x) + " y: " + str(y)

    def testtocsrVCSC(self, VCSCMatrix, SPMatrix):
        # print(VCSCMatrix)

        csr_from_vcsc = VCSCMatrix.tocsr()


        result = csr_from_vcsc - SPMatrix

        assert epsilon > abs(result.sum()), "resultvcsc: " + str(csr_from_vcsc.sum())
        
        for x in range(SPMatrix.shape[0]):
            for y in range(SPMatrix.shape[1]):
                assert epsilon > abs(SPMatrix[x, y] - csr_from_vcsc[x, y]), "SPMatrix: " + str(SPMatrix[x, y]) + " csc_from_vcsc: " + str(csr_from_vcsc[x, y]) + " x: " + str(x) + " y: " + str(y)


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
        VCSCMatrix_copy = VCSCMatrix.copy()
        IVCSCMatrix_copy = IVCSCMatrix.copy()

        del VCSCMatrix
        del IVCSCMatrix

        VCSCMatrix = vcsc.VCSC(SPMatrix)
        IVCSCMatrix = ivcsc.IVCSC(SPMatrix)

        assert epsilon > abs(VCSCMatrix_copy.sum() - VCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())
        assert epsilon > abs(IVCSCMatrix_copy.sum() - IVCSCMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " IVCSCMatrix: " + str(IVCSCMatrix.sum())

        VCSC_CSC_Copy = VCSCMatrix_copy.tocsc()
        IVCSC_CSC_Copy = IVCSCMatrix_copy.tocsc()

        VCSC_CSC = VCSCMatrix.tocsc()
        IVCSC_CSC = IVCSCMatrix.tocsc()

        result = (VCSC_CSC_Copy - VCSC_CSC).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[1])), decimal=2, verbose=True)

        result = (IVCSC_CSC_Copy - IVCSC_CSC).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[1])), decimal=2, verbose=True)


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

        if(SPMatrix.format == "csr"):
            IVCSC_check = VCSCMatrix.tocsr()
        else:
            IVCSC_check = VCSCMatrix.tocsc()
        
        result = (IVCSC_check - SPMatrix).toarray()
        np.testing.assert_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[1])), decimal=2, verbose=True)


    def testScalarMultiplyVCSC(self, SPMatrix, VCSCMatrix):
        VCSCresult = VCSCMatrix * 2
        SPresult = SPMatrix * 2
        
        if(SPMatrix.format == "csr"):
            IVCSC_check = VCSCresult.tocsr()
        else:
            IVCSC_check = VCSCresult.tocsc()
        
        result = (IVCSC_check - SPresult).toarray()
        np.testing.assert_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[1])), decimal=2, verbose=True)



    def testIPScalarMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCMatrix *= 2
        SPMatrix *= 2
        
        if(SPMatrix.format == "csr"):
            IVCSC_check = IVCSCMatrix.tocsr()
        else:
            IVCSC_check = IVCSCMatrix.tocsc()
        
        result = (IVCSC_check - SPMatrix).toarray()
        np.testing.assert_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[1])), decimal=2, verbose=True)


    def testScalarMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCresult = IVCSCMatrix * 2
        SPresult = SPMatrix * 2

        if(SPMatrix.format == "csr"):
            IVCSC_check = IVCSCresult.tocsr()
        else:
            IVCSC_check = IVCSCresult.tocsc()
        
        result = (IVCSC_check - SPresult).toarray()
        np.testing.assert_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[1])), decimal=2, verbose=True)



    def testMatrixMultiplyVCSC(self, SPMatrix, VCSCMatrix):
        VCSCresult = VCSCMatrix *  SPMatrix.transpose().toarray()
        SPresult = SPMatrix *  SPMatrix.transpose().toarray()

        # assert epsilon > abs(VCSCresult.sum() - SPresult.sum()), "VCSCresult: " + str(VCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
    
        result = (VCSCresult - SPresult) # SHOULD be a matrix of all zeros
        np.testing.assert_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[0])), decimal=2, verbose=True)

    def testMatrixMultiplyIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCresult = IVCSCMatrix * SPMatrix.transpose().toarray()
        SPresult = SPMatrix * SPMatrix.transpose().toarray()

        # assert epsilon > abs(IVCSCresult.sum() - SPresult.sum()), "IVCSCresult: " + str(IVCSCresult.sum()) + " SPresult: " + str(SPresult.sum())
        result = (IVCSCresult - SPresult) # SHOULD be a matrix of all zeros
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix.shape[0], SPMatrix.shape[0])), decimal=2, verbose=True)


    def testAppendOwnFormat(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        VCSCMat2 = vcsc.VCSC(SPMatrix)
        IVCSCMat2 = ivcsc.IVCSC(SPMatrix)   
        VCSCMat2.append(VCSCMatrix)
        IVCSCMat2.append(IVCSCMatrix)        

        VCSC2CSC = VCSCMat2.tocsc()
        IVCSC2CSC = IVCSCMat2.tocsc()

        result = (VCSC2CSC - IVCSC2CSC).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((result.shape[0], result.shape[1])), decimal=2, verbose=True)
    
    def testTrace(self, SPMatrix, VCSCMatrix, IVCSCMatrix):
        if SPMatrix.shape[0] != SPMatrix.shape[1]:
            with pytest.raises(ValueError):
                VCSCMatrix.trace()
        else:
            assert epsilon > abs(VCSCMatrix.trace() -  IVCSCMatrix.trace()), "VCSCMatrix: " + str(VCSCMatrix.trace()) + " IVCSCMatrix: " + str(IVCSCMatrix.trace())
            assert epsilon > abs(VCSCMatrix.trace() - SPMatrix.trace()), "VCSCMatrix: " + str(VCSCMatrix.trace()) + " IVCSCMatrix: " + str(IVCSCMatrix.trace()) + " SPMatrix: " + str(SPMatrix.trace())

    
    def testAppendCSC_VCSC(self, SPMatrix):
        if SPMatrix.format == "csr":
            pytest.skip("Skipping append CSC test because SPMatrix is a csr matrix")
        SPMatrix3 = sp.sparse.hstack([SPMatrix, SPMatrix]).A
        VCSCMat = vcsc.VCSC(SPMatrix)   
        VCSCMat.append(SPMatrix)
        VCSC2CSC = VCSCMat.tocsc()
        result = (SPMatrix3 - VCSC2CSC).A
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix3.shape[0], SPMatrix3.shape[1])), decimal=2)
        assert epsilon > abs(result.sum()), "VCSCMat: " + str(VCSCMat.sum()) + " SPMatrix3: " + str(SPMatrix3.sum())
    

    def testAppendCSC_IVCSC(self, SPMatrix):
        if SPMatrix.format == "csr":
            pytest.skip("Skipping append CSC test because SPMatrix is a csr matrix")
        SPMatrix3 = sp.sparse.hstack([SPMatrix, SPMatrix]).A
        IVCSCMat = ivcsc.IVCSC(SPMatrix)   
        IVCSCMat.append(SPMatrix)
        IVCSC2CSC = IVCSCMat.tocsc()
        result = (SPMatrix3 - IVCSC2CSC).A
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix3.shape[0], SPMatrix3.shape[1])), decimal=2)
        assert epsilon > abs(result.sum()), "IVCSCMat: " + str(IVCSCMat.sum()) + " SPMatrix3: " + str(SPMatrix3.sum())
    
    
    def testAppendCSR_VCSC(self, SPMatrix):
        if SPMatrix.format == "csc":
            pytest.skip("Skipping append CSR test because SPMatrix is a CSC matrix")
        SPMatrix3 = sp.sparse.vstack([SPMatrix, SPMatrix]).A
        VCSCMat = vcsc.VCSC(SPMatrix)   
        VCSCMat.append(SPMatrix)
        VCSC2CSC = VCSCMat.tocsr()
        result = (SPMatrix3 - VCSC2CSC).A
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix3.shape[0], SPMatrix3.shape[1])), decimal=2)
        assert epsilon > abs(result.sum()), "VCSCMat: " + str(VCSCMat.sum()) + " SPMatrix3: " + str(SPMatrix3.sum())
    

    def testAppendCSR_IVCSC(self, SPMatrix):
        if SPMatrix.format == "csc":
            pytest.skip("Skipping append CSR test because SPMatrix is a CSC matrix")
        SPMatrix3 = sp.sparse.vstack([SPMatrix, SPMatrix]).A
        IVCSCMat = ivcsc.IVCSC(SPMatrix)   
        IVCSCMat.append(SPMatrix)
        IVCSC2CSC = IVCSCMat.tocsr()
        result = (SPMatrix3 - IVCSC2CSC).A
        
        np.testing.assert_array_almost_equal(result, np.zeros((SPMatrix3.shape[0], SPMatrix3.shape[1])), decimal=2)
        assert epsilon > abs(result.sum()), "IVCSCMat: " + str(IVCSCMat.sum()) + " SPMatrix3: " + str(SPMatrix3.sum())
    
    def testAppendWrongFormat(self, VCSCMatrix, IVCSCMatrix):
        with pytest.raises(TypeError):
            VCSCMatrix.append(IVCSCMatrix)
        with pytest.raises(TypeError):
            IVCSCMatrix.append(VCSCMatrix)



    def test_getValues(self, VCSCMatrix, SPMatrix):

        sum = 0
        for col in range(VCSCMatrix.outerSize):

            valueArray = VCSCMatrix.getValues(col)
            countsArray = VCSCMatrix.getCounts(col)

            for i in range(len(valueArray)):
                sum += valueArray[i] * countsArray[i]
        
        assert abs(sum - SPMatrix.sum()) < epsilon

    def test_getIndicesCOLUMN(self, VCSCMatrix, SPMatrix):
        if self.format == "csr":
            pytest.skip("Skipping column test for csr")

        values = []
        indices = []
        counts = []

        for col in range(SPMatrix.shape[1]):
            values  = VCSCMatrix.getValues(col)
            indices = VCSCMatrix.getIndices(col)
            counts  = VCSCMatrix.getCounts(col)

            indexCounter = 0
            for count in counts:
                i = 0
                while i < count:
                    assert SPMatrix[indices[indexCounter], col] == values[0]
                    i += 1
                    indexCounter += 1
                values = values[1:]

    def test_getIndicesROW(self, VCSCMatrix, SPMatrix):
            if self.format == "csc":
                pytest.skip("Skipping row test for csc")
            
            values = []
            indices = []
            counts = []
    
            for row in range(SPMatrix.shape[0]):
                values  = VCSCMatrix.getValues(row)
                indices = VCSCMatrix.getIndices(row)
                counts  = VCSCMatrix.getCounts(row)
                
                indexCounter = 0
                for count in counts:
                    i = 0
                    while i < count:
                        assert SPMatrix[row, indices[indexCounter]] == values[0]
                        i += 1
                        indexCounter += 1
                    values = values[1:]

    def test_getIndicesBadInput(self, VCSCMatrix, SPMatrix):
        with pytest.raises(IndexError):
            VCSCMatrix.getIndices(VCSCMatrix.outerSize + 1)
    
    def test_getValuesBadInput(self, VCSCMatrix, SPMatrix):
        with pytest.raises(IndexError):
            VCSCMatrix.getValues(VCSCMatrix.outerSize + 1)
    
    def test_getCountsBadInput(self, VCSCMatrix, SPMatrix):
        with pytest.raises(IndexError):
            VCSCMatrix.getCounts(VCSCMatrix.outerSize + 1)

    def test_getNumIndicesBadInput(self, VCSCMatrix, SPMatrix):
        with pytest.raises(IndexError):
            VCSCMatrix.getNumIndices(VCSCMatrix.outerSize + 1)

    def test_getIndicesNegativeInput(self, VCSCMatrix):
        
        for i in range(VCSCMatrix.outerSize - 1):
            negI = i - VCSCMatrix.outerSize
            forward = VCSCMatrix.getIndices(i)
            backward = VCSCMatrix.getIndices(negI)

            for x in range(len(forward)):
                assert forward[x] == backward[x]
            

    def test_getValuesNegativeInput(self, VCSCMatrix):
            
        for i in range(VCSCMatrix.outerSize - 1):
            negI = i - VCSCMatrix.outerSize
            forward = VCSCMatrix.getValues(i)
            backward = VCSCMatrix.getValues(negI)

            for x in range(len(forward)):
                assert forward[x] == backward[x]

    def test_getCountsNegativeInput(self, VCSCMatrix):

        for i in range(VCSCMatrix.outerSize - 1):
            negI = i - VCSCMatrix.outerSize
            forward = VCSCMatrix.getCounts(i)
            backward = VCSCMatrix.getCounts(negI)

            for x in range(len(forward)):
                assert forward[x] == backward[x]

    def test_getNumIndicesNegativeInput(self, VCSCMatrix):
            
        for i in range(VCSCMatrix.outerSize - 1):
            negI = i - VCSCMatrix.outerSize
            forward = VCSCMatrix.getNumIndices(i)
            backward = VCSCMatrix.getNumIndices(negI)

            assert forward == backward
    
    
    def test_VCSC_to_CSC_Constructor(self, VCSCMatrix):
        VCSCMatrix2 = vcsc.VCSC(VCSCMatrix)
        assert epsilon > abs(VCSCMatrix.sum() - VCSCMatrix2.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " VCSCMatrix2: " + str(VCSCMatrix2.sum())
        assert VCSCMatrix.shape() == VCSCMatrix2.shape(), "VCSCMatrix: " + str(VCSCMatrix.shape()) + " VCSCMatrix2: " + str(VCSCMatrix2.shape())

        CSC_Copy = VCSCMatrix.tocsc()
        CSC2_Copy = VCSCMatrix2.tocsc()

        result = (CSC_Copy - CSC2_Copy).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((VCSCMatrix.shape()[0], VCSCMatrix.shape()[1])), decimal=2, verbose=True)

    def test_IVCSC_to_CSC_Constructor(self, IVCSCMatrix):
        IVCSCMatrix2 = ivcsc.IVCSC(IVCSCMatrix)
        assert epsilon > abs(IVCSCMatrix.sum() - IVCSCMatrix2.sum()), "IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " IVCSCMatrix2: " + str(IVCSCMatrix2.sum())
        assert IVCSCMatrix.shape() == IVCSCMatrix2.shape(), "IVCSCMatrix: " + str(IVCSCMatrix.shape()) + " IVCSCMatrix2: " + str(IVCSCMatrix2.shape())

        CSC_Copy = IVCSCMatrix.tocsc()
        CSC2_Copy = IVCSCMatrix2.tocsc()

        result = (CSC_Copy - CSC2_Copy).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((IVCSCMatrix.shape()[0], IVCSCMatrix.shape()[1])), decimal=2, verbose=True)
    
    def test_VCSC_COO_Constructor(self, SPMatrix):
        COO = sp.sparse.coo_matrix(SPMatrix)
        VCSCMatrix = vcsc.VCSC(COO)
        assert epsilon > abs(VCSCMatrix.sum() - SPMatrix.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())
        assert VCSCMatrix.shape() == SPMatrix.shape, "VCSCMatrix: " + str(VCSCMatrix.shape()) + " SPMatrix: " + str(SPMatrix.shape)

        CSC_Copy = VCSCMatrix.tocsc()
        CSC2_Copy = SPMatrix.tocsc()

        result = (CSC_Copy - CSC2_Copy).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((VCSCMatrix.shape()[0], VCSCMatrix.shape()[1])), decimal=2, verbose=True)

    def test_IVCSC_COO_Constructor(self, SPMatrix):
        COO = sp.sparse.coo_matrix(SPMatrix)
        IVCSCMatrix = ivcsc.IVCSC(COO)
        assert epsilon > abs(IVCSCMatrix.sum() - SPMatrix.sum()), "IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " SPMatrix: " + str(SPMatrix.sum())
        assert IVCSCMatrix.shape() == SPMatrix.shape, "IVCSCMatrix: " + str(IVCSCMatrix.shape()) + " SPMatrix: " + str(SPMatrix.shape)

        CSC_Copy = IVCSCMatrix.tocsc()
        CSC2_Copy = SPMatrix.tocsc()

        result = (CSC_Copy - CSC2_Copy).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((IVCSCMatrix.shape()[0], IVCSCMatrix.shape()[1])), decimal=2, verbose=True)

    def testInnerSumVCSC(self, SPMatrix, VCSCMatrix):
        vcsc_sum = VCSCMatrix.sum(axis=1)
        sp_sum = SPMatrix.sum(axis=1,dtype=VCSCMatrix.dtype)

        np.testing.assert_almost_equal(vcsc_sum, sp_sum, decimal=3)
    
    def testInnerSumIVCSC(self, SPMatrix, IVCSCMatrix):
        ivcsc_sum = IVCSCMatrix.sum(axis=1)
        sp_sum = SPMatrix.sum(axis=1,dtype=IVCSCMatrix.dtype)

        np.testing.assert_almost_equal(ivcsc_sum, sp_sum, decimal=3)

    def testMaxColVCSC(self, SPMatrix, VCSCMatrix):
        vcsc_max = VCSCMatrix.max(axis=0)
        sp_max = SPMatrix.max(axis=0).toarray()
        np.testing.assert_almost_equal(vcsc_max, sp_max, decimal=3)

    def testMaxColIVCSC(self, SPMatrix, IVCSCMatrix):
        ivcsc_max = IVCSCMatrix.max(axis=0)
        sp_max = SPMatrix.max(axis=0).toarray()
        np.testing.assert_almost_equal(ivcsc_max, sp_max, decimal=3)

    def testMaxValVCSC(self, SPMatrix, VCSCMatrix):
        vcsc_max = VCSCMatrix.max()
        sp_max = SPMatrix.max()
        np.testing.assert_almost_equal(vcsc_max, sp_max, decimal=3)

    def testMaxValIVCSC(self, SPMatrix, IVCSCMatrix):
        ivcsc_max = IVCSCMatrix.max()
        sp_max = SPMatrix.max()
        np.testing.assert_almost_equal(ivcsc_max, sp_max, decimal=3)

    def testMaxRowVCSC(self, SPMatrix, VCSCMatrix):
        vcsc_max = VCSCMatrix.max(axis=1)
        sp_max = SPMatrix.max(axis=1).toarray()
        np.testing.assert_almost_equal(vcsc_max, sp_max, decimal=3)

    def testMaxRowIVCSC(self, SPMatrix, IVCSCMatrix):
        ivcsc_max = IVCSCMatrix.max(axis=1)
        sp_max = SPMatrix.max(axis=1).toarray()
        np.testing.assert_almost_equal(ivcsc_max, sp_max, decimal=3)

    def testMinValVCSC(self, SPMatrix, VCSCMatrix):
        vcsc_min = VCSCMatrix.min()
        sp_min = SPMatrix.min()

        np.testing.assert_almost_equal(vcsc_min, sp_min, decimal=3)

    def testMinValIVCSC(self, SPMatrix, IVCSCMatrix):
        ivcsc_min = IVCSCMatrix.min()
        sp_min = SPMatrix.min()

        np.testing.assert_almost_equal(ivcsc_min, sp_min, decimal=3)

    def testOuterSumVCSC(self, SPMatrix, VCSCMatrix):
        vcsc_sum = VCSCMatrix.sum(axis=0)
        sp_sum = SPMatrix.sum(axis=0, dtype=VCSCMatrix.dtype)
        np.testing.assert_almost_equal(vcsc_sum, sp_sum, decimal=3)

    def testOuterSumIVCSC(self, SPMatrix, IVCSCMatrix):
        ivcsc_sum = IVCSCMatrix.sum(axis=0)
        sp_sum = SPMatrix.sum(axis=0,dtype=IVCSCMatrix.dtype)
        np.testing.assert_almost_equal(ivcsc_sum, sp_sum, decimal=3)


    def testConstrctVCSCFromVCSC(self, VCSCMatrix):
        vcsc2 = vcsc.VCSC(VCSCMatrix)
        assert epsilon > abs(VCSCMatrix.sum() - vcsc2.sum()), "VCSCMatrix: " + str(VCSCMatrix.sum()) + " vcsc2: " + str(vcsc2.sum())
        assert VCSCMatrix.shape() == vcsc2.shape(), "VCSCMatrix: " + str(VCSCMatrix.shape()) + " vcsc2: " + str(vcsc2.shape())

        CSC_Copy = VCSCMatrix.tocsc()
        CSC2_Copy = vcsc2.tocsc()

        result = (CSC_Copy - CSC2_Copy).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((VCSCMatrix.shape()[0], VCSCMatrix.shape()[1])), decimal=2, verbose=True)

    def testConstrctIVCSCFromIVCSC(self, IVCSCMatrix):
        ivcsc2 = ivcsc.IVCSC(IVCSCMatrix)
        assert epsilon > abs(IVCSCMatrix.sum() - ivcsc2.sum()), "IVCSCMatrix: " + str(IVCSCMatrix.sum()) + " ivcsc2: " + str(ivcsc2.sum())
        assert IVCSCMatrix.shape() == ivcsc2.shape(), "IVCSCMatrix: " + str(IVCSCMatrix.shape()) + " ivcsc2: " + str(ivcsc2.shape())

        CSC_Copy = IVCSCMatrix.tocsc()
        CSC2_Copy = ivcsc2.tocsc()

        result = (CSC_Copy - CSC2_Copy).toarray()
        np.testing.assert_array_almost_equal(result, np.zeros((IVCSCMatrix.shape()[0], IVCSCMatrix.shape()[1])), decimal=2, verbose=True)

    def testRandomAccessVCSC(self, SPMatrix, VCSCMatrix):
        for x in range(SPMatrix.shape[0]):
            for y in range(SPMatrix.shape[1]):
                assert VCSCMatrix[x, y] == SPMatrix[x, y]

    def testRandomAccessIVCSC(self, SPMatrix, IVCSCMatrix):
        for x in range(SPMatrix.shape[0]):
            for y in range(SPMatrix.shape[1]):
                assert IVCSCMatrix[x, y] == SPMatrix[x, y]

    def testReadFromNPZVCSC(self, SPMatrix):
        try:
            sp.sparse.save_npz("test.npz", SPMatrix)
            newVCSC = vcsc.VCSC("test.npz")
        finally:
            os.remove("test.npz")
        assert np.allclose(newVCSC.tocsc().toarray(), SPMatrix.toarray(), atol=epsilon)

    def testReadFromNPZIVCSC(self, SPMatrix):
        try:
            sp.sparse.save_npz("test.npz", SPMatrix)
            newIVCSC = ivcsc.IVCSC("test.npz")
        finally:
            os.remove("test.npz")
        assert np.allclose(newIVCSC.tocsc().toarray(), SPMatrix.toarray(), atol=epsilon)

    def testReadFromNPZ_bad_format_VCSC(self, SPMatrix):
        try:
            bsrMat = SPMatrix.tobsr()
            sp.sparse.save_npz("bsr.npz", bsrMat)
            newVCSC = vcsc.VCSC("bsr.npz")
        finally:
            os.remove("bsr.npz")
        assert np.allclose(newVCSC.tocsc().toarray(), SPMatrix.toarray(), atol=epsilon)

    def testReadFromNPZ_bad_format_IVCSC(self, SPMatrix):
        try:
            bsrMat = SPMatrix.tobsr()
            sp.sparse.save_npz("bsr.npz", bsrMat)
            newIVCSC = ivcsc.IVCSC("bsr.npz")
        finally:
            os.remove("bsr.npz")
        assert np.allclose(newIVCSC.tocsc().toarray(), SPMatrix.toarray(), atol=epsilon)


    def testWriteReadVCSC(self, VCSCMatrix):
        VCSCMatrix.write("test.vcsc")

        try:
            result = vcsc.VCSC("test.vcsc")
        finally:
            os.remove("test.vcsc")
            
        originalCSC = VCSCMatrix.tocsc()
        resultCSC = result.tocsc()

        np.testing.assert_array_almost_equal(originalCSC.toarray(), resultCSC.toarray(), decimal=3, verbose=True)

    def testWriteReadIVCSC(self, IVCSCMatrix):
        IVCSCMatrix.write("test.ivcsc")
        
        try:
            result = ivcsc.IVCSC("test.ivcsc")
        finally:
            os.remove("test.ivcsc")
        
        originalCSC = IVCSCMatrix.tocsc()
        resultCSC = result.tocsc()

        np.testing.assert_array_almost_equal(originalCSC.toarray(), resultCSC.toarray(), decimal=3, verbose=True)
        
    def testWriteReadDifferenDataTypeVCSC(self, SPMatrix, VCSCMatrix):
        VCSCMatrix.write("test.vcsc")
        
        if SPMatrix.dtype == np.float32:
            temp = SPMatrix.astype(np.float64)
        else:
            temp = SPMatrix.astype(np.float32)


        vcsc2 = vcsc.VCSC(temp)
        try:
            vcsc2.read("test.vcsc")
        finally:
            os.remove("test.vcsc")
        
        originalCSC = VCSCMatrix.tocsc()
        resultCSC = vcsc2.tocsc()

        np.testing.assert_array_almost_equal(originalCSC.toarray(), resultCSC.toarray(), decimal=3, verbose=True)

    def testWriteReadDifferenDataTypeIVCSC(self, SPMatrix, IVCSCMatrix):
        IVCSCMatrix.write("test.ivcsc")
        
        if SPMatrix.dtype == np.float32 or SPMatrix.dtype == np.float64:
            temp = SPMatrix.astype(np.int32)
        else:
            temp = SPMatrix.astype(np.float32)

        ivcsc2 = ivcsc.IVCSC(temp)

        try:
            ivcsc2.read("test.ivcsc")
        finally:
            os.remove("test.ivcsc")
        
        originalCSC = IVCSCMatrix.tocsc()
        resultCSC = ivcsc2.tocsc()

        np.testing.assert_array_almost_equal(originalCSC.toarray(), resultCSC.toarray(), decimal=3, verbose=True)
