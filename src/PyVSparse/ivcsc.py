from __future__ import annotations

import scipy as sp
import numpy as np

import PyVSparse


class IVCSC:
    def __init__(self, spmat, order: str = "col"): # add scipySparseMat: scipyFormat as type hint

        self.order = order.lower().capitalize()
        self.dtype: np.dtype = spmat.dtype
        self.format = "ivcsc"

        if(spmat.nnz == 0):
            raise ValueError("Cannot construct IVCSC from empty matrix")

        if self.order != "Col" and self.order != "Row":
            raise TypeError("storage order must be one of: 'Col', 'Row'")

        self.rows: np.uint32 = np.uint32(0)
        self.cols: np.uint32 = np.uint32(0)
        self.nnz: np.uint32 = np.uint32(0)
        self.innerSize: np.uint32 = np.uint32(0)
        self.outerSize: np.uint32 = np.uint32(0)
        self.bytes: np.uint64 = np.uint64(0)

        if(spmat.nnz == 0):
            raise ValueError("Cannot construct VCSC from empty matrix")
        if(spmat.format == "csc"):
            self.order = "Col"
            moduleName = "PyVSparse._PyVSparse._IVCSC._" + str(self.dtype) + "_" + str(self.order)
            self._CSconstruct(moduleName, spmat)
        elif(spmat.format == "csr"):
            self.order = "Row"
            moduleName = "PyVSparse._PyVSparse._IVCSC._" + str(self.dtype) + "_" + str(self.order)
            self._CSconstruct(moduleName, spmat)    
        elif(spmat.format == "coo"):
            moduleName = "PyVSparse._PyVSparse._IVCSC._" + str(self.dtype) + "_" + str(self.order)
            self._COOconstruct(moduleName, spmat)
        elif(isinstance(spmat, IVCSC)): # TODO test
            self.fromIVCSC(spmat)
        elif(isinstance(spmat, PyVSparse.VCSC)): #TODO test
            self.fromVCSC(spmat)
        else:
            raise TypeError("Input matrix does not have a valid format!")
        
    def fromIVCSC(self, ivcsc: IVCSC):
        self.wrappedForm = ivcsc.wrappedForm.copy()
        self.dtype = ivcsc.dtype
        self.indexT = ivcsc.indexT
        self.rows = ivcsc.rows
        self.cols = ivcsc.cols
        self.nnz = ivcsc.nnz
        self.innerSize = ivcsc.innerSize
        self.outerSize = ivcsc.outerSize
        self.bytes = ivcsc.bytes

    def fromVCSC(self, vcscMat: PyVSparse.VCSC):
        raise NotImplementedError
        
    def copy(self):
        return IVCSC(self)

    def __repr__(self) -> None:
        self.wrappedForm.print()

    def __str__(self) -> str:
        return self.wrappedForm.__str__()

    def __deepcopy__(self, memo): # https://stackoverflow.com/a/46939443/12895299
        _copy = self.copy()
        return _copy

    def sum(self) -> int:
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
    
    def norm(self) -> np.double: 
        return self.wrappedForm.norm()
    
    def byteSize(self) -> np.uint64: 
        return self.wrappedForm.byteSize
    
    def vectorLength(self, vector) -> np.double: # TODO fix
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

    def transpose(self, inplace = True): # -> IVCSC:
        return self.wrappedForm.transpose()

    def shape(self) -> tuple[np.uint32, np.uint32]: 
        return (self.rows, self.cols)
    
    def __imul__(self, other: np.ndarray) -> IVCSC:

        if(isinstance(other, int) or isinstance(other, float)):
            self.wrappedForm.__imul__(other)
        else:
            raise TypeError("Cannot multiply IVCSC by " + str(type(other)))
            
        return self
    
    def __mul__(self, other: np.ndarray):

        if(isinstance(other, np.ndarray)):
            temp = self.wrappedForm.__mul__(other)
            return temp
        elif(isinstance(other, int) or isinstance(other, float)):
            result = self
            result.wrappedForm = self.wrappedForm.__mul__(other)
            return result
        else:
            raise TypeError("Cannot multiply IVCSC by " + str(type(other)))
            
    def __eq__(self, other) -> bool:
        return self.wrappedForm.__eq__(other)
    
    def __ne__(self, other) -> bool:
        return self.wrappedForm.__ne__(other)

    # def __iter__(self):
        # return self.wrappedForm.__iter__()
    
    def getValues(self) -> list[int]: # TODO test
        return self.wrappedForm.getValues()
    
    def getIndices(self) -> list[int]: # TODO test
        return self.wrappedForm.getIndices()
    
    def append(self, matrix) -> None: # TODO fix

        if isinstance(matrix, IVCSC) and self.order == matrix.order:
            self.wrappedForm.append(matrix.wrappedForm)
            self.rows += matrix.shape()[0] # type: ignore 
            self.cols += matrix.shape()[1] # type: ignore 
        elif isinstance(matrix, sp.sparse.csc_matrix) and self.order == "Col":
            self.wrappedForm.append(matrix)
            self.rows += matrix.shape[0] # type: ignore
            self.cols += matrix.shape[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csr_matrix) and self.order == "Row":
            self.wrappedForm.append(matrix)
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


    def slice(self, start, end) -> IVCSC: # TODO fix
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
        self.indexT: np.dtype = type(spmat.indices[0])
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
