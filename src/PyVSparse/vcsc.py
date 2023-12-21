from __future__ import annotations
from typing import overload

import scipy as sp
import numpy as np

from PyVSparse.ivcsc import IVCSC
import PyVSparse 

class VCSC:

    def __init__(self, spmat, major: str = "col", indexT: np.dtype = np.dtype(np.uint32)):
        if(spmat.nnz == 0):
            raise ValueError("Cannot construct VCSC from empty matrix")

        self.major = major.lower().capitalize()
        self.dtype: np.dtype = spmat.dtype
        self.indexT = np.dtype(indexT) 
    
        if(isinstance(self.indexT, type(np.dtype(np.uint32)))):
            self.indexT = np.uint32
        elif(isinstance(self.indexT, type(np.dtype(np.uint64)))):
            self.indexT = np.uint64
        elif(isinstance(self.indexT, type(np.dtype(np.uint16)))):
            self.indexT = np.uint16
        elif(isinstance(self.indexT, type(np.dtype(np.uint8)))):
            self.indexT = np.uint8
        else:
            raise TypeError("indexT must be one of: np.uint8, np.uint16, np.uint32, np.uint64")

        if self.major != "Col" and self.major != "Row":
            raise TypeError("major must be one of: 'Col', 'Row'")

        if(spmat.format == "csc"):
            self.major = "Col"
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexT)) + "_" + str(self.major)
            self._CSconstruct(moduleName, spmat)
        elif(spmat.format == "csr"):
            self.major = "Row"
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexT)) + "_" + str(self.major)
            self._CSconstruct(moduleName, spmat)    
        elif(spmat.format == "coo"):
            moduleName = "PyVSparse._PyVSparse._VCSC." + str(self.dtype) + "_" + str(np.dtype(self.indexT)) + "_" + str(self.major)    
            self._COOconstruct(moduleName, spmat)
        elif(isinstance(spmat, VCSC)): # TODO test
            self = spmat

    def fromPyVSparse(self, vcsc: VCSC):
        self.wrappedForm = vcsc.wrappedForm
        self.dtype = vcsc.dtype
        self.indexT = vcsc.indexT
        self.rows = vcsc.rows
        self.cols = vcsc.cols
        self.nnz = vcsc.nnz
        self.inner = vcsc.inner
        self.outer = vcsc.outer
        self.bytes = vcsc.byteSize()

    
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
        
    def sum(self) -> int: # tested
        return self.wrappedForm.sum()

    def trace(self) -> int: # TODO fix
        return self.wrappedForm.trace()

    def outerSum(self) -> list[int]: # TODO test
        return self.wrappedForm.outerSum()

    def innerSum(self) -> list[int]: # TODO test
        return self.wrappedForm.innerSum()
    
    def maxColCoeff(self) -> list[int]: # TODO test
        return self.wrappedForm.maxColCoeff()
    
    def maxRowCoeff(self) -> list[int]: # TODO test
        return self.wrappedForm.maxRowCoeff()

    def minColCoeff(self) -> list[int]: # TODO test
        return self.wrappedForm.minColCoeff()
    
    def minRowCoeff(self) -> list[int]: # TODO test
        return self.wrappedForm.minRowCoeff()

    def byteSize(self) -> np.uint64: # TODO test
        return self.wrappedForm.byteSize
    
    def norm(self) -> np.double: 
        return self.wrappedForm.norm()
    
    def vectorLength(self, vector) -> np.double: # TODO test
        return self.wrappedForm.vectorLength(vector)

    def tocsc(self) -> sp.sparse.csc_matrix:
        if self.major == "Row":
            return self.wrappedForm.toEigen().tocsc()
        return self.wrappedForm.toEigen()
    
    def tocsr(self) -> sp.sparse.csr_matrix:
        if self.major == "Col":
            return self.tocsc().tocsr()
        else:
            return self.wrappedForm.toEigen()

    def transpose(self, inplace = True) -> VCSC:
        if inplace:
            self.wrappedForm = self.wrappedForm.transpose()
            self.rows, self.cols = self.cols, self.rows
            self.inner, self.outer = self.outer, self.inner
            return self
        temp = self
        temp.wrappedForm = self.wrappedForm.transpose()
        temp.rows, temp.cols = self.cols, self.rows
        temp.inner, temp.outer = self.outer, self.inner
        return temp
        
    

    def shape(self) -> tuple[np.uint32, np.uint32]: # TODO test
        return (self.rows, self.cols)
    
    def __imul__(self, other: np.ndarray) -> VCSC: 

        if(type(other) == int or type(other) == float):
            self.wrappedForm.__imul__(other)
        else:
            raise TypeError("Cannot multiply VCSC by " + str(type(other)))
            
        return self
    
    def __mul__(self, other):

        if(isinstance(other, np.ndarray)):
            temp: np.ndarray = self.wrappedForm * other
            return temp
        elif(isinstance(other, int) or isinstance(other, float)):
            result = self
            result.wrappedForm = self.wrappedForm * other
            return result
        else:
            raise TypeError("Cannot multiply VCSC by " + str(type(other)))
            
    def __eq__(self, other) -> bool:
        return self.wrappedForm.__eq__(other)
    
    def __ne__(self, other) -> bool:
        return self.wrappedForm.__ne__(other)
    
    def getValues(self) -> list[int]: # TODO test
        return self.wrappedForm.getValues()
    
    def getIndices(self) -> list[int]: # TODO test
        return self.wrappedForm.getIndices()
    
    def getCounts(self) -> list[int]: # TODO test
        return self.wrappedForm.getCounts()
    
    def getNumIndices(self) -> list[int]: # TODO test
        return self.wrappedForm.getNumIndices()
    
    def append(self, matrix) -> None: # TODO fix

        if isinstance(matrix, VCSC) and self.major == matrix.major:
            self.wrappedForm.append(matrix.wrappedForm)
            self.rows += matrix.shape()[0] # type: ignore
            self.cols += matrix.shape()[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csc_matrix) and self.major == "Col":
            self.wrappedForm.append(matrix)
            self.rows += matrix.shape[0] # type: ignore
            self.cols += matrix.shape[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csr_matrix) and self.major == "Row":
            self.wrappedForm.append(matrix.tocsc())
            self.rows += matrix.shape[0] # type: ignore
            self.cols += matrix.shape[1] # type: ignore
        else:
            raise TypeError("Cannot append " + str(type(matrix)) + " to " + str(type(self)))

        self.nnz += matrix.nnz

        if self.major == "Col":
            self.inner += self.rows
            self.outer += self.cols
        else:
            self.inner += self.cols
            self.outer += self.rows



    def slice(self, start, end) -> VCSC:  # TODO fix
        result = self
        result.wrappedForm = self.wrappedForm.slice(start, end)
        result.nnz = result.wrappedForm.nnz

        if(self.major == "Col"):
            result.inner = self.rows
            result.outer = end - start
            result.cols = result.outer
            result.rows = self.rows
        else:
            result.inner = self.cols
            result.outer = end - start
            result.rows = result.outer
            result.cols = self.cols

        return result

    def _CSconstruct(self, moduleName: str, spmat):
        self.indexT = type(spmat.indices[0])
        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz
        self.inner: np.uint32 = spmat.indices
        self.outer: np.uint32 = spmat.indptr
        # print("Constructing VCSC with moduleName: ", moduleName)
        self.wrappedForm = eval(str(moduleName))(spmat)
        self.bytes: np.uint64 = self.wrappedForm.byteSize

    def _COOconstruct(self, moduleName: str, spmat): # TODO test
        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz
        
        if(self.major == "Col"):
            self.inner: np.uint32 = spmat.row
            self.outer: np.uint32 = spmat.col
        else:
            self.inner: np.uint32 = spmat.col
            self.outer: np.uint32 = spmat.row

        self.wrappedForm = eval(str(moduleName))((spmat.row, spmat.col, spmat.data), self.rows, self.cols, spmat.nnz)
        self.bytes: np.uint64 = self.wrappedForm.byteSize
