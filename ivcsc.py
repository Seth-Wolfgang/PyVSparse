import PyVSparse
import scipy as sp
import numpy as np

class IVCSC:
    def __init__(self, scipySparseMat):

        if scipySparseMat.format == "csc": 
            self.major = "Col" 
        else:
            self.major = "Row"

        self.indexT = type(scipySparseMat.indices[0])
        self.dtype = scipySparseMat.dtype
        self.rows = scipySparseMat.shape[0]
        self.cols = scipySparseMat.shape[1]
        self.shape = scipySparseMat.shape
        self.inner = scipySparseMat.indices
        self.outer = scipySparseMat.indptr
        self.wrappedForm = eval(str("PyVSparse.IVCSC__" + self._CDTypeConvert(self.dtype) + "_uint64_t_" + str(self.major)))(scipySparseMat)
        self.byteSize = self.wrappedForm.byteSize


    def __repr__(self):
        self.wrappedForm.print()

    def sum(self):
        return self.wrappedForm.sum()

    def trace(self):
        return self.wrappedForm.trace()

    def outerSum(self):
        return self.wrappedForm.outerSum()

    def innerSum(self):
        return self.wrappedForm.innerSum()
    
    def maxColCoeff(self):
        return self.wrappedForm.maxColCoeff()
    
    def maxRowCoeff(self):
        return self.wrappedForm.maxRowCoeff()

    def minColCoeff(self):
        return self.wrappedForm.minColCoeff()
    
    def minRowCoeff(self):
        return self.wrappedForm.minRowCoeff()
    
    def norm(self):
        return self.wrappedForm.norm()
    
    def vectorLength(self, vector):
        return self.wrappedForm.vectorLength(vector)

    def toSciPySparse(self):
        return self.wrappedForm.toEigen()

    def transpose(self, inplace = True):
        return self.wrappedForm.transpose()
    
    def __imul__(self, other):
        return self.wrappedForm.__imul__(other)
    
    def __eq__(self, other):
        return self.wrappedForm.__eq__(other)
    
    def __ne__(self, other):
        return self.wrappedForm.__ne__(other)
    
    def getValues(self):
        return self.wrappedForm.getValues()
    
    def getIndices(self):
        return self.wrappedForm.getIndices()
    
    def getCounts(self):
        return self.wrappedForm.getCounts()
    
    def getNumIndices(self):
        return self.wrappedForm.getNumIndices()
    
    def append(self, matrix):
        self.wrappedForm.append(matrix)
    

    def _CDTypeConvert(self, dtype: np.dtype) -> str:
        match dtype:
            case np.int8:
                return "int8_t"
            case np.int16:
                return "int16_t"
            case np.int32:
                return "int32_t"
            case np.int64:
                return "int64_t"
            case np.uint8:
                return "uint8_t"
            case np.uint16:
                return "uint16_t"
            case np.uint32:
                return "uint32_t"
            case np.uint64:
                return "uint64_t"
            case np.float32:
                return "float"
            case np.float64:
                return "double"
        return "unknown"