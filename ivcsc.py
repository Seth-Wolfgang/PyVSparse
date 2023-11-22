from typing import TypeVar
import PyVSparse
import scipy as sp
import numpy as np

# scipyFormats = TypeVar("scipyFormats", sp.sparse.csr_matrix, sp.sparse.csc_matrix, sp.sparse.coo_matrix)

class IVCSC:
    def __init__(self, scipySparseMat): # add scipySparseMat: scipyFormat as type hint

        if scipySparseMat.format == "csc": 
            self.major = "Col" 
        else:
            self.major = "Row"

        self.indexT = type(scipySparseMat.indices[0])
        self.dtype = scipySparseMat.dtype
        self.rows: int = scipySparseMat.shape[0]
        self.cols: int = scipySparseMat.shape[1]
        self.shape = scipySparseMat.shape
        self.inner = scipySparseMat.indices
        self.outer = scipySparseMat.indptr
        self.wrappedForm = eval(str("PyVSparse.IVCSC_" + self._CDTypeConvert(self.dtype) + "_uint64_t_" + str(self.major)))(scipySparseMat)
        self.byteSize = self.wrappedForm.byteSize

    # def __init__(self, IVSparseMat: IVCSC):



    def __repr__(self) -> None:
        self.wrappedForm.print()

    def __str__(self) -> str:
        return self.wrappedForm.__str__()

    def sum(self) -> int:
        return self.wrappedForm.sum()

    def trace(self) -> int:
        return self.wrappedForm.trace()

    def outerSum(self) -> list[int]:
        return self.wrappedForm.outerSum()

    def innerSum(self) -> list[int]:
        return self.wrappedForm.innerSum()
    
    def maxColCoeff(self) -> list[int]:
        return self.wrappedForm.maxColCoeff()
    
    def maxRowCoeff(self) -> list[int]:
        return self.wrappedForm.maxRowCoeff()

    def minColCoeff(self) -> list[int]:
        return self.wrappedForm.minColCoeff()
    
    def minRowCoeff(self) -> list[int]:
        return self.wrappedForm.minRowCoeff()
    
    def norm(self) -> np.double:
        return self.wrappedForm.norm()
    
    def vectorLength(self, vector) -> np.double:
        return self.wrappedForm.vectorLength(vector)

    def toCSC(self):
        return self.wrappedForm.toEigen()

    def transpose(self, inplace = True): # -> IVCSC:
        return self.wrappedForm.transpose()

    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)
    
    def __imul__(self, other):
        self.wrappedForm.__imul__(other)
    
    def __eq__(self, other) -> bool:
        return self.wrappedForm.__eq__(other)
    
    def __ne__(self, other) -> bool:
        return self.wrappedForm.__ne__(other)

    # def __iter__(self):
        # return self.wrappedForm.__iter__()
    
    def getValues(self) -> list[int]:
        return self.wrappedForm.getValues()
    
    def getIndices(self) -> list[int]:
        return self.wrappedForm.getIndices()
    
    def getCounts(self) -> list[int]:
        return self.wrappedForm.getCounts()
    
    def getNumIndices(self) -> list[int]:
        return self.wrappedForm.getNumIndices()
    
    def append(self, matrix) -> None:
        self.wrappedForm.append(matrix)

    def slice(self, start, end): #-> IVCSC:
        return self.wrappedForm.slice(start, end)
    
    #TODO find a better method of doing this
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