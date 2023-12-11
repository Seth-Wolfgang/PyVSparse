from __future__ import annotations
from typing import overload
import PyVSparse._VCSC
import scipy as sp
import numpy as np

from ivcsc import IVCSC

class VCSC:

    def __init__(self, spmat, major: str = "col", indexT: np.dtype = np.dtype(np.uint32)):
        if(spmat.nnz == 0):
            raise ValueError("Cannot construct VCSC from empty matrix")

        self.major = major.lower().capitalize()
        self.dtype: np.dtype = spmat.dtype
        self.indexT: np.dtype = indexT
        
        if(spmat.format == "csc"):
            self.major = "Col"
            moduleName = "PyVSparse._VCSC._" + str(self.dtype) + "_" + str(self.indexT) + "_" + str(self.major)
            self._CSconstruct(moduleName, spmat)
        elif(spmat.format == "csr"):
            self.major = "Row"
            moduleName = "PyVSparse._VCSC." + str(self.dtype) + "_" + str(self.indexT) + "_" + str(self.major)
            self._CSconstruct(moduleName, spmat)    
        elif(spmat.format == "coo"):
            moduleName = "PyVSparse._VCSC." + str(self.dtype) + "_" + str(self.indexT) + "_" + str(self.major)    
            self._COOconstruct(moduleName, spmat)
        elif(hasattr(spmat, "wrappedForm")):
            self = spmat

    def fromPyVSparse(self, vcsc: VCSC):
        self.wrappedForm = vcsc.wrappedForm
        self.dtype = vcsc.dtype
        self.indexT = vcsc.indexT
        self.rows = vcsc.rows
        self.cols = vcsc.cols
        self.nnz = vcsc.nnz
        self.shape = vcsc.shape
        self.inner = vcsc.inner
        self.outer = vcsc.outer
        self.byteSize = vcsc.byteSize

    
    def __repr__(self):
        return self.wrappedForm.__repr__()

    def __str__(self) -> str:
        return self.wrappedForm.__str__()

    # def __iter__(self, outerIndex: int):
    #     self.iter = self.wrappedForm.__iter__(outerIndex)
    #     return self.iter
    
    # def __next__(self):    
    #     if(self.iter):
    #         return self.iter.__next__()
    #     else:
    #         raise StopIteration
        
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

    def toCSC(self) -> sp.sparse.csc_matrix:
        return self.wrappedForm.toEigen()
    
    # def toCSR(self) -> sp.sparse.csr_matrix:
    #     return self.toCSC().toCSR()
        # return self.wrappedForm.toEigen()

    def transpose(self, inplace = True) -> VCSC:
        if inplace:
            self.wrappedForm = self.wrappedForm.transpose()
            self.rows, self.cols = self.cols, self.rows
            self.shape = (self.rows, self.cols)
            self.inner, self.outer = self.outer, self.inner
            return
        temp = self
        temp.wrappedForm = self.wrappedForm.transpose()
        temp.rows, temp.cols = self.cols, self.rows
        temp.shape = (self.rows, self.cols)
        temp.inner, temp.outer = self.outer, self.inner
        return temp
        
    
    def __imul__(self, other):
        self.wrappedForm.__imul__(other)
        return self

    def __mul__(self, other: np.ndarray) -> np.ndarray:
        return  self.wrappedForm.__mul__(other)

    def __mul__(self, other: int) -> VCSC:
        self.wrappedForm = self.wrappedForm.__mul__(other)
        return self

    def __eq__(self, other) -> bool:
        return self.wrappedForm.__eq__(other)
    
    def __ne__(self, other) -> bool:
        return self.wrappedForm.__ne__(other)
    
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

    def slice(self, start, end) -> VCSC: 
        result = self
        result.wrappedForm = self.wrappedForm.slice(start, end)

        return result

    def _CSconstruct(self, moduleName: str, spmat):
        self.indexT = type(spmat.indices[0])
        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz
        self.shape = spmat.shape
        self.inner: np.uint32 = spmat.indices
        self.outer: np.uint32 = spmat.indptr
        # print("Constructing VCSC with moduleName: ", moduleName)
        self.wrappedForm = eval(str(moduleName))(spmat)
        self.byteSize: np.uint64 = self.wrappedForm.byteSize

    def _COOconstruct(self, moduleName: str, spmat):
        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz
        self.shape = spmat.shape
        
        if(self.major == "Col"):
            self.inner: np.uint32 = spmat.row
            self.outer: np.uint32 = spmat.col
        else:
            self.inner: np.uint32 = spmat.col
            self.outer: np.uint32 = spmat.row

        self.wrappedForm = eval(str(moduleName))((spmat.row, spmat.col, spmat.data), self.rows, self.cols, spmat.nnz)
        self.byteSize: np.uint64 = self.wrappedForm.byteSize
