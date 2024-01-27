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

        """
        Copy constructor for VCSC
        """

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
        

    def sum(self, axis=None):

        """
        On axis=None, returns the sum of all elements in the matrix

        If axis=0, returns the sum of each column

        If axis=1, returns the sum of each row

        Note: Sum is either int64 or a double
        """
        if axis is None:
            return self.wrappedForm.sum()
        if axis == 0:
            return self.wrappedForm.colSum()
        if axis == 1:
            return self.wrappedForm.rowSum()
        

    def trace(self): 
        
        """
        Returns the sum of all elements along the diagonal. 

        Throws ValueError if matrix is not square.
        
        Note: Sum is either int64 or a double.

        """

        if self.rows != self.cols:
            raise ValueError("Cannot take trace of non-square matrix")
        
        return self.wrappedForm.trace()
    
    def max(self, axis=None):
            
        """
        On axis=None, returns the maximum of all elements in the matrix

        If axis=0, returns the maximum of each column

        If axis=1, returns the maximum of each row

        """
        if axis is None:
            return self.wrappedForm.max()
        else:
            return self.wrappedForm.max(axis)
    
    def min(self, axis=None):
                
        """
        On axis=None, returns the minimum of all *nonzero* elements in the matrix 

        If axis=0, returns the *nonzero* minimum of each column

        If axis=1, returns the *nonzero*  minimum of each row

        Note: because of the way the matrix is stored, 
              minimums that are zero are very expensive to compute.
              
              There are a few exceptions: 
              - If a row/column is all zeros, then the minimum will be zero.
              - if axis=None, then the minimum will be zero if nnz < rows * cols

        """
        if axis is None:
            return self.wrappedForm.min()
        else:
            return self.wrappedForm.min(axis)

    def byteSize(self) -> np.uint64: 
        """
        Returns the memory consumption of the matrix in bytes
        """

        return self.wrappedForm.byteSize
    
    def norm(self) -> np.double: 
        
        """
        Returns the Frobenius norm of the matrix
        """
        
        return self.wrappedForm.norm()
    
    def vectorLength(self, vector) -> np.double: # TODO test
        
        """
        Returns the magnitude of the vector
        """

        return self.wrappedForm.vectorLength(vector)

    def tocsc(self) -> sp.sparse.csc_matrix:

        """
        Converts the matrix to a scipy.sparse.csc_matrix

        Note: This is a copy. This does not destroy the original matrix.
        """

        if self.order == "Row":
            return self.wrappedForm.toEigen().tocsc()
        return self.wrappedForm.toEigen()
    
    def tocsr(self) -> sp.sparse.csr_matrix:

        """
        Converts the matrix to a scipy.sparse.csr_matrix

        Note: This is a copy. This does not destroy the original matrix.
        """

        if self.order == "Col":
            return self.tocsc().tocsr()
        else:
            return self.wrappedForm.toEigen()

    def transpose(self, inplace = True) -> VCSC:
        
        """
        Transposes the matrix.

        Note: This is a very slow operation. It is recommended to use the transpose() function from scipy.sparse.csc_matrix instead.
        """
        
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

        if(isinstance(other, np.ndarray)): # Dense numpy matrix or vector
            temp: np.ndarray = self.wrappedForm * other
            return temp
        elif(isinstance(other, int) or isinstance(other, float)): # Scalar
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

        """
        Returns the unique values of a column or row
        
        Note: Whether the values are from a column or row depends on order of the matrix.
              A matrix stored in column-major order will return the values of a column.
        """

        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getValues(outerIndex)
    
    def getIndices(self, outerIndex: int) -> list: 

        """
        Returns the indices of a column or row

        Note: Whether the indices are from a column or row depends on order of the matrix.
                A matrix stored in column-major order will return the indices of a column.
        """

        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getIndices(outerIndex)
    
    def getCounts(self, outerIndex: int) -> list:

        """
        Returns the number of non-zero elements in a column or row

        For example, if the matrix is:
            [1] 
            [1]
            [2]
        Then the list [1, 2] will be returned

        Note: Whether the counts are from a column or row depends on order of the matrix.
                A matrix stored in column-major order will return the counts of a column.
        """

        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getCounts(outerIndex)
    
    def getNumIndices(self, outerIndex: int) -> list: 
        
        """
        Returns the number of unique values in a column or row

        Note: Whether the number of indices are from a column or row depends on order of the matrix.
                A matrix stored in column-major order will return the number of indices of a column.
        """
        
        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.wrappedForm.getNumIndices(outerIndex)
    
    def append(self, matrix) -> None: 

        """
        Appends a matrix to the current matrix

        The appended matrix must be of the same type or a scipy.sparse.csc_matrix/csr_matrix 
        depending on the storage order of the current matrix. For a column-major matrix,
        the appended matrix will be appended to the end of the columns. For a row-major matrix,
        the appended matrix will be appended to the end of the rows.
        """

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

        """
        Returns a slice of the matrix.

        Currently, only slicing by storage order is supported. For example, if the matrix is stored in column-major order,
        Then the returned matrix will be a slice of the columns.
        """

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
