from __future__ import annotations

import scipy as sp
import numpy as np

import PyVSparse


class IVCSC:
    def __init__(self, spmat, order: str = "col"): # add scipySparseMat: scipyFormat as type hint

        self.order = order.lower().capitalize()
        self.format = "ivcsc"

        self.rows: np.uint32 = np.uint32(0)
        self.cols: np.uint32 = np.uint32(0)
        self.nnz: np.uint32 = np.uint32(0)
        self.innerSize: np.uint32 = np.uint32(0)
        self.outerSize: np.uint32 = np.uint32(0)
        self.bytes: np.uint64 = np.uint64(0)
        self.dtype: np.dtype = np.dtype(None)
        

        if self.order != "Col" and self.order != "Row":
            raise TypeError("storage order must be one of: 'Col', 'Row'")

        # If the input is a scipy.sparse matrix or IVSparse matrix
        if not isinstance(spmat, str):
            self.dtype: np.dtype = spmat.dtype

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
        elif isinstance(spmat, str):
            self.read(spmat)
        else:
            raise TypeError("Input matrix must be a string or a scipy.sparse matrix")
        
    def fromIVCSC(self, ivcsc: IVCSC):

        """
        Copy constructor for IVCSC
        """

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
    
    def norm(self) -> np.double: # todo: make this not just Froebenius norm

        """
        Returns the Frobenius norm of the matrix
        """

        return self.wrappedForm.norm()
    
    def byteSize(self) -> np.uint64: 
        """
        Returns the memory consumption of the matrix in bytes
        """

        return self.wrappedForm.byteSize
    
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

    def transpose(self, inplace = True) -> IVCSC:
        
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

    def append(self, matrix) -> None: 
        """
        Appends a matrix to the current matrix

        The appended matrix must be of the same type or a scipy.sparse.csc_matrix/csr_matrix 
        depending on the storage order of the current matrix. For a column-major matrix,
        the appended matrix will be appended to the end of the columns. For a row-major matrix,
        the appended matrix will be appended to the end of the rows.
        """

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


    def slice(self, start, end) -> IVCSC: 

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
    
    def write(self, filename: str) -> None: 

        """
        Writes the matrix to a file. IVCSC does not have a file
        extension, so the user does not need to specify a extension.
        """

        self.wrappedForm.write(filename)

  

    # def read(self, filename: str, dtype=None, order=None):

    #     """
    #     This reads a IVCSC formatted matrix from a file. 
        
    #     The user can specify the dtype and order of the matrix.
    #     Each one MUST be specified or the function will default to 
    #     assuming the matrix is the same type as the current matrix AND 
    #     initialized.

    #     If the matrix is not initialized, then the function will throw an error.
    #     """
    #     if self.wrappedForm is None and (dtype is None or order is None):
    #         raise ValueError("Template data must be specified for an uninitialized matrix.")
    #     elif dtype is None or order is None:
    #         try:
    #             self.wrappedForm = self.wrappedForm.read(filename)
    #         except:
    #             raise IOError("Could not open file: " + filename + ". Check that the file is in the correct format or written from IVCSC.write().")
    #     else:
    #         try:
    #             self.wrappedForm = eval("PyVSparse._PyVSparse._IVCSC._" + str(self.dtype) + "_" + str(self.order))(filename)
    #         except:
    #             raise IOError("Could not open file: " + filename + ". Check that the file is in the correct format or written from IVCSC.write().")
    #         self.dtype = dtype
    #         self.order = order

    #     self.rows = self.wrappedForm.rows
    #     self.cols = self.wrappedForm.cols
    #     self.nnz = self.wrappedForm.nonZeros()
    #     self.innerSize = self.wrappedForm.innerSize
    #     self.outerSize = self.wrappedForm.outerSize
    #     self.bytes = self.wrappedForm.byteSize

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

    def read(self, filename: str):

        """
        Private helper function to read a VCSC formatted matrix from a file.
        This function should automatically determine the template type of the matrix.
        """

        assert filename[-6:] == ".ivcsc", "File must have a .ivcsc extension"

        try:
            matFile = open(filename, "rb")
        except:
            raise IOError("Could not open file: " + filename + ". Check that the file is in the correct format or written from IVCSC.write().")

        matFile.seek(16)
        valueByte = matFile.read(4)
        matFile.close()

        typeSize = int(valueByte[0])
        isFloating = bool(valueByte[1])
        isSigned = bool(valueByte[2])
        isColumnMajor = bool(valueByte[3])
        
        if isFloating:
            if typeSize == 4:
                self.dtype = np.dtype(np.float32)
            elif typeSize == 8:
                self.dtype = np.dtype(np.float64)
            else:
                raise ValueError(f"Invalid floating point flag byte in IVCSC file: {filename}. Value: {valueByte}")
        else:
            if typeSize == 1:
                self.dtype = np.dtype(np.int8) if isSigned else np.dtype(np.uint8)
            elif typeSize == 2:
                self.dtype = np.dtype(np.int16) if isSigned else np.dtype(np.uint16)
            elif typeSize == 4:
                self.dtype = np.dtype(np.int32) if isSigned else np.dtype(np.uint32)
            elif typeSize == 8:
                self.dtype = np.dtype(np.int64) if isSigned else np.dtype(np.uint64)
            else:
                raise ValueError(f"Invalid type size in IVCSC file: {filename}. Value: {typeSize}")

        if isColumnMajor:
            self.order = "Col"
        else:
            self.order = "Row"

        self.wrappedForm = eval(str("PyVSparse._PyVSparse._IVCSC._" + str(self.dtype) + "_" + str(self.order)))(filename)
        # try:
        # self.wrappedForm.read()
        # except:
            # raise IOError("Could not open file: " + filename + ". Check that the file is in the correct format or written from IVCSC.write().")

        self.rows = self.wrappedForm.rows
        self.cols = self.wrappedForm.cols
        self.nnz = self.wrappedForm.nonZeros()
        self.innerSize = self.wrappedForm.innerSize
        self.outerSize = self.wrappedForm.outerSize
        self.bytes = self.wrappedForm.byteSize

