from __future__ import annotations

import scipy as sp
import numpy as np

import PyVSparse 

class VCSC:

    def __init__(self, spmat, order: str = "col", indexT: np.dtype = np.dtype(np.uint32)):
        if(spmat.nnz == 0):
            raise ValueError("Cannot construct VCSC from empty matrix")

        self.order = order.lower().capitalize()
        self.dtype: np.dtype = spmat.dtype
        self.indexT = np.dtype(indexT) 
        self.format = "vcsc"

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

        if self.order != "Col" and self.order != "Row":
            raise TypeError("major must be one of: 'Col', 'Row'")

        self.rows: np.uint32 = np.uint32(0)
        self.cols: np.uint32 = np.uint32(0)
        self.nnz: np.uint64 = np.uint64(0)
        self.innerSize: np.uint32 = np.uint32(0)
        self.outerSize: np.uint32 = np.uint32(0)
        self.bytes: np.uint64 = np.uint64(0)

        if(spmat.format == "csc"):
            self.order = "Col"
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexT)) + "_" + str(self.order)
            self._CSconstruct(moduleName, spmat)
        elif(spmat.format == "csr"):
            self.order = "Row"
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexT)) + "_" + str(self.order)
            self._CSconstruct(moduleName, spmat)    
        elif(spmat.format == "coo"):
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexT)) + "_" + str(self.order)    
            self._COOconstruct(moduleName, spmat)
        elif(isinstance(spmat, VCSC)): # TODO test
            self.fromVCSC(spmat)
        elif(isinstance(spmat, PyVSparse.IVCSC)): #TODO test
            self.fromIVCSC(spmat)
        else:
            raise TypeError("Input matrix does not have a valid format!")


    def fromVCSC(self, spmat: VCSC):
        self.wrappedForm = spmat.wrappedForm.copy()
        self.dtype = spmat.dtype
        self.indexT = spmat.indexT
        self.rows = spmat.rows
        self.cols = spmat.cols
        self.nnz = spmat.nnz
        self.innerSize = spmat.innerSize
        self.outerSize = spmat.outerSize
        self.bytes = spmat.byteSize()

    def fromIVCSC(self, spmat: PyVSparse.IVCSC):
        raise NotImplementedError
    
    def __repr__(self):
        return self.wrappedForm.__repr__()

    def __str__(self) -> str:
        return self.wrappedForm.__str__()

    def __deepcopy__(self): 
        _copy = VCSC(self)
        return _copy

    def copy(self):
        return VCSC(self)

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

    def trace(self): 
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

    def byteSize(self) -> np.uint64: 
        return self.wrappedForm.byteSize
    
    def norm(self) -> np.double: 
        return self.wrappedForm.norm()
    
    def vectorLength(self, vector) -> np.double: # TODO test
        return self.wrappedForm.vectorLength(vector)

    def tocsc(self) -> sp.sparse.csc_matrix:
        if self.order == "Row":
            return self.wrappedForm.toEigen().tocsc()
        return self.wrappedForm.toEigen()
    
    def tocsr(self) -> sp.sparse.csr_matrix:
        if self.order == "Col":
            return self.tocsc().tocsr()
        else:
            return self.wrappedForm.toEigen()

    def transpose(self, inplace = True) -> VCSC:
        if inplace:
            self.wrappedForm = self.wrappedForm.transpose()
            self.rows, self.cols = self.cols, self.rows
            self.innerSize, self.outerSize = self.outerSize, self.innerSize
            return self
        temp = self
        temp.wrappedForm = self.wrappedForm.transpose()
        temp.rows, temp.cols = self.cols, self.rows
        temp.innerSize, temp.outerSize = self.outerSize, self.innerSize
        return temp
        
    

    def shape(self) -> tuple[np.uint32, np.uint32]: 
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
    
    def getValues(self, outerIndex: int) -> list: 
        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getValues(outerIndex)
    
    def getIndices(self, outerIndex: int) -> list: 
        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getIndices(outerIndex)
    
    def getCounts(self, outerIndex: int) -> list: 
        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getCounts(outerIndex)
    
    def getNumIndices(self, outerIndex: int) -> list: 
        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getNumIndices(outerIndex)
    
    def append(self, matrix) -> None: # TODO fix

        if isinstance(matrix, VCSC) and self.order == matrix.order:
            self.wrappedForm.append(matrix.wrappedForm)
            self.rows += matrix.shape()[0] # type: ignore
            self.cols += matrix.shape()[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csc_matrix) and self.order == "Col":
            self.wrappedForm.append(matrix)
            self.rows += matrix.shape[0] # type: ignore
            self.cols += matrix.shape[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csr_matrix) and self.order == "Row":
            self.wrappedForm.append(matrix.tocsc())
            self.rows += matrix.shape[0] # type: ignore
            self.cols += matrix.shape[1] # type: ignore
        else:
            raise TypeError("Cannot append " + str(type(matrix)) + " to " + str(type(self)))

        self.nnz += matrix.nnz # type: ignore

        if self.order == "Col":
            self.innerSize += self.rows
            self.outerSize += self.cols
        else:
            self.innerSize += self.cols
            self.outerSize += self.rows



    def slice(self, start, end) -> VCSC:  # TODO fix
        result = self
        result.wrappedForm = self.wrappedForm.slice(start, end)
        result.nnz = result.wrappedForm.nonZeros()

        if(self.order == "Col"):
            result.innerSize = self.rows
            result.outerSize = end - start
            result.cols = result.outerSize
            result.rows = self.rows
        else:
            result.innerSize = self.cols
            result.outerSize = end - start
            result.rows = result.outerSize
            result.cols = self.cols

        return result

    def _CSconstruct(self, moduleName: str, spmat):
        self.indexT = type(spmat.indices[0])
        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz

        if(self.order == "Col"):
            self.innerSize: np.uint32 = self.rows
            self.outerSize: np.uint32 = self.cols
        else:
            self.innerSize: np.uint32 = self.cols
            self.outerSize: np.uint32 = self.rows

        self.wrappedForm = eval(str(moduleName))(spmat)
        self.bytes: np.uint64 = self.wrappedForm.byteSize

    def _COOconstruct(self, moduleName: str, spmat): 
        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz
        
        if(self.order == "Col"):
            self.innerSize: np.uint32 = spmat.row
            self.outerSize: np.uint32 = spmat.col
        else:
            self.innerSize: np.uint32 = spmat.col
            self.outerSize: np.uint32 = spmat.row
        
        coords = []
        for r, c, v in zip(spmat.row, spmat.col, spmat.data):
            coords.append((r, c, v))    

        self.wrappedForm = eval(str(moduleName))(coords, self.rows, self.cols, spmat.nnz)
        self.bytes: np.uint64 = self.wrappedForm.byteSize
