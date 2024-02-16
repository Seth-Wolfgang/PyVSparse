from __future__ import annotations

import scipy as sp
import numpy as np

import PyVSparse 

class VCSC:

    def __init__(self, spmat, indexType = np.uint32, order: str = "col"):

        """
        Value-Compressed Sparse Column is a read-only sparse matrix format for redundant data without compromising speed. 
        See README.md for more information. 
        
        While the name is indicitive of the storage order, the matrix can be stored in either column-major or row-major order.

        This class can be constructed from a few different options:
        1.) scipy.sparse.csc_matrix
        2.) scipy.sparse.csr_matrix
        3.) scipy.sparse.coo_matrix
        4.) PyVSparse.VCSC
        5.) A .vcsc file (written by VCSC.write() or a C++ program that uses VCSC.write()).

        The user can specify the index type of the matrix. The default is np.uint32. The python version is
        limited to unsigned integers, becasue there is no advantage to using signed integers. The choice of
        index type should not affect performance, correctness, or what features are available. The only difference
        SHOULD be the memory consumption of the matrix, unless you are attempting to store more than the integer
        limit of the index type. i.e. storing 256 rows/cols with a np.uint8 index type.

        The user can also specify the storage order of the matrix. The default is "Col" for column-major order.

        Note: Becasue of how indices are stored in VCSC, cache misses are more commmon. For a very redundant matrix,
              the performance of VCSC will be just as fast, in some cases faster, than a CSC matrix because caching is possible.
              However, this is based on naive implementation of matrix operations, so matrix multiplication will be faster for 
              SciPy matrices that use BLAS. 

              VCSC is faster than IVCSC becasue indices are byte-aligned, but does not offer the same level of compression. 

              Coefficient-wise operations may be much faster because fewer are stored.

        :param spmat: The input matrix or .vcsc filename
        :type spmat: Union[sp.sparse.csc_matrix, sp.sparse.csr_matrix, sp.sparse.coo_matrix, PyVSparse.VCSC, str]

        :param indexType: The index type of the matrix. The default is np.uint32
        :type indexType: np.dtype

        :param order: The storage order of the matrix. The default is "Col" for column-major order. "Row" can also be specified for row-major order. Capitalization does not matter.
        :type order: str
        """

        self.order = order.lower().capitalize()
        self.indexType = np.dtype(indexType) 
        self.format: str = "vcsc"

        if(isinstance(self.indexType, type(np.dtype(np.uint32)))):
            self.indexType = np.uint32
        elif(isinstance(self.indexType, type(np.dtype(np.uint64)))):
            self.indexType = np.uint64
        elif(isinstance(self.indexType, type(np.dtype(np.uint16)))):
            self.indexType = np.uint16
        elif(isinstance(self.indexType, type(np.dtype(np.uint8)))):
            self.indexType = np.uint8
        else:
            raise TypeError("indexType must be one of: np.uint8, np.uint16, np.uint32, np.uint64")

        if self.order != "Col" and self.order != "Row":
            raise TypeError("major must be one of: 'Col', 'Row'")

        self.rows: np.uint32 = np.uint32(0)
        self.cols: np.uint32 = np.uint32(0)
        self.nnz: np.uint64 = np.uint64(0)
        self.innerSize: np.uint32 = np.uint32(0) # For column-major, this is the number of rows. For row-major, this is the number of columns.
        self.outerSize: np.uint32 = np.uint32(0) # For column-major, this is the number of columns. For row-major, this is the number of rows.
        self.bytes: np.uint64 = np.uint64(0)
        self.dtype: np.dtype = np.dtype(None)
        

        # If the input is a scipy.sparse matrix or IVSparse matrix
        if not isinstance(spmat, str):
            self.dtype: np.dtype = spmat.dtype
            if spmat.nnz == 0:
                raise ValueError("Input matrix must have at least one non-zero element")
            
            if(spmat.format == "csc"):
                self.order = "Col"

                # Because IVSparse is a templated C++ library, the compiled C++ backend has many different types.
                # So a string is constructed to call the correct C++ constructor.
                moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)
                self._CSconstruct(moduleName, spmat)
            elif(spmat.format == "csr"):
                self.order = "Row"
                moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)
                self._CSconstruct(moduleName, spmat)    
            elif(spmat.format == "coo"):
                moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)    
                self._COOconstruct(moduleName, spmat)
            elif(isinstance(spmat, VCSC)): # TODO test
                self.fromVCSC(spmat)
            elif(isinstance(spmat, PyVSparse.IVCSC)): #TODO test
                self.fromIVCSC(spmat)
            else:
                raise TypeError("Input matrix does not have a valid format!")
        elif isinstance(spmat, str):
            if(spmat[-4:] == ".npz"):
                self._npzConstruct(spmat)
            elif(spmat[-5:] == ".vcsc"):
                self.read(spmat)
            else:
                raise TypeError("Input must be a .vcsc or a scipy.sparse .npz file.")
        else:
            raise TypeError("Input must be a filename or a scipy.sparse matrix")


    def fromVCSC(self, spmat: VCSC):

        """
        Copy constructor for VCSC

        :param spmat: The input VCSC matrix
        :type spmat: VCSC
        """

        self.backend = spmat.backend.copy()
        self.dtype = spmat.dtype
        self.indexType = spmat.indexType
        self.rows = spmat.rows
        self.cols = spmat.cols
        self.nnz = spmat.nnz
        self.innerSize = spmat.innerSize
        self.outerSize = spmat.outerSize
        self.bytes = spmat.byteSize()

    def fromIVCSC(self, spmat: PyVSparse.IVCSC):
        raise NotImplementedError
    
    def __repr__(self):
        return self.backend.__repr__()

    def __str__(self) -> str:
        return self.backend.__str__()

    def __deepcopy__(self): 
        _copy = VCSC(self)
        return _copy

    def copy(self):
        return VCSC(self)
        

    def sum(self, axis=None):

        """
        On axis=None, returns the sum of all elements in the matrix

        If axis=0, returns the sum of each column

        If axis=1, returns the sum of each row

        Note: Sum is either int64 or a double

        :param axis: The axis to sum along. The default is None, which sums all elements in the matrix.
        :type axis: int

        :return: The sum of the matrix or the sum of each row/column
        :rtype: Union[np.int64, np.double, np.ndarray]
        :raises ValueError: If the axis is not 0, 1, or None
        """

        if axis is None:
            return self.backend.sum()
        elif axis == 0:
            return self.backend.colSum()
        elif axis == 1:
            return self.backend.rowSum()
        else:
            raise ValueError("Axis must be 0, 1, or None")
        

    def trace(self): 
        
        """
        Returns the sum of all elements along the diagonal. 

        Throws ValueError if matrix is not square.
        
        Note: Sum is either int64 or a double.

        :return: The sum of the diagonal
        :rtype: Union[np.int64, np.double]
        :raises ValueError: If the matrix is not square
        """

        if self.rows != self.cols:
            raise ValueError("Cannot take trace of non-square matrix")
        
        return self.backend.trace()
    
    def max(self, axis=None):
            
        """
        On axis=None, returns the maximum of all elements in the matrix

        If axis=0, returns the maximum of each column

        If axis=1, returns the maximum of each row

        :param axis: The axis to find the maximum along. The default is None, which finds the maximum of all elements in the matrix.
        :type axis: int
        :return: The maximum of the matrix or the maximum of each row/column
        :rtype: Union[np.int64, np.double, np.ndarray]
        """

        if axis is None:
            return self.backend.max()
        else:
            return self.backend.max(axis)
    
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
            
        :param axis: The axis to find the minimum along. The default is None, which finds the minimum of all *nonzero* elements in the matrix.
        :type axis: int
        :return: The minimum of the matrix or the minimum of each row/column
        :rtype: Union[np.int64, np.double, np.ndarray]
        """

        if axis is None:
            return self.backend.min()
        else:
            return self.backend.min(axis)

    def byteSize(self) -> np.uint64: 
        """
        Returns the memory consumption of the matrix in bytes
        
        :rtype: np.uint64
        """

        return self.backend.byteSize
    
    def norm(self) -> np.double: # TODO add more norms
        
        """
        Returns the Frobenius norm of the matrix

        :return: The Frobenius norm of the matrix
        :rtype: np.double
        """
        
        return self.backend.norm()
    
    def vectorLength(self, vector: int) -> np.double: # TODO test
        
        """
        Returns the euclidean length of the vector
        
        :param vector: The index of the vector to find the euclidean length of
        :type vector: int
        :return: The euclidean length of the vector
        :rtype: np.double
        :raises IndexError: If the vector index is out of range
        """

        if vector > self.outerSize:
            raise IndexError("Vector index out of range")
        elif vector < 0:
            vector += self.outerSize

        return self.backend.vectorLength(vector)

    def tocsc(self) -> sp.sparse.csc_matrix:

        """
        Converts the matrix to a scipy.sparse.csc_matrix

        Note: This is a copy. This does not destroy the original matrix.
              If the storage order of the VCSC matrix is in row-major, then
              then a csr_matrix will be created and converted to a csc_matrix.
        
        :return: The matrix in csc format
        :rtype: sp.sparse.csc_matrix
        """

        if self.order == "Row":
            return self.backend.toEigen().tocsc()
        return self.backend.toEigen()
    
    def tocsr(self) -> sp.sparse.csr_matrix:

        """
        Converts the matrix to a scipy.sparse.csr_matrix

        Note: This is a copy. This does not destroy the original matrix.
              If the storage order of the VCSC matrix is in column-major, then 
              then a csc_matrix will be created and converted to a csr_matrix.
        
        :return: The matrix in scipy.sparse.csr_matrix format
        :rtype: sp.sparse.csr_matrix
        """

        if self.order == "Col":
            return self.tocsc().tocsr()
        else:
            return self.backend.toEigen()

    def transpose(self, inplace = True) -> VCSC:
        
        """
        Transposes the matrix.

        Note: This is a very slow operation. It is recommended to use the transpose() function from another matrix format instead.
              Nothing is returned if the operation is in place.
        
              Memory usage will change after this operation.
              
        :param inplace: Whether to transpose the matrix in place. The default is True
        :type inplace: bool

        :return: The transposed VCSC matrix
        :rtype: VCSC
        """
        
        if inplace:
            self.backend = self.backend.transpose()
            self.rows, self.cols = self.cols, self.rows
            self.innerSize, self.outerSize = self.outerSize, self.innerSize
            self.bytes = self.backend.byteSize
            return self
        temp = self
        temp.backend = self.backend.transpose()
        temp.rows, temp.cols = self.cols, self.rows
        temp.innerSize, temp.outerSize = self.outerSize, self.innerSize
        temp.bytes = temp.backend.byteSize
        return temp
        
    

    def shape(self) -> tuple[np.uint32, np.uint32]: 
        """
        Returns the shape of the matrix as a tuple (rows, cols)

        :return The shape of the matrix
        :rtype: Tuple[np.uint32, np.uint32]
        """

        return (self.rows, self.cols)
    
    def __imul__(self, other) -> VCSC: 

        """
        Inplace multiplication of the matrix by a scalar

        :param other: The value or object to multiply the matrix by
        :type other: Union[int, float]
        :return: The matrix multiplied by the input
        :rtype: VCSC
        :raises TypeError: If the input is not a scalar or numpy array      
        """

        if(type(other) == int or type(other) == float):
            self.backend.__imul__(other)
        else:
            raise TypeError("Cannot multiply VCSC by " + str(type(other)))
            
        return self
    
    def __mul__(self, other):

        """
        Multiplication of the matrix by a:
        - scalar
        - dense numpy matrix or vector

        If the input is a scalar, then the matrix returned will be a VCSC matrix.
        Else, the matrix returned will be a dense numpy matrix or vector.

        :param other: The value or object to multiply the matrix by
        :type other: Union[int, float, np.ndarray]
        :return: The matrix multiplied by the input
        :rtype: union[VCSC, np.ndarray]
        :raises TypeError: If the input is not a scalar or numpy array
        """

        if(isinstance(other, np.ndarray)): # Dense numpy matrix or vector
            temp: np.ndarray = self.backend * other
            return temp
        elif(isinstance(other, int) or isinstance(other, float)): # Scalar
            result = self
            result.backend = self.backend * other
            return result
        else:
            raise TypeError("Cannot multiply VCSC by " + str(type(other)))
            
    def __eq__(self, other: VCSC) -> bool:

        """
        Compares the matrix to another VCSC matrix

        :param other: The matrix to compare to
        :type other: VCSC
        :return: True if the matrices are equal, False otherwise
        :rtype: bool
        """

        return self.backend.__eq__(other)
    
    def __ne__(self, other: VCSC) -> bool:
        
        """
        Compares the matrix to another VCSC matrix

        :param other: The matrix to compare to
        :type other: VCSC
        :return: True if the matrices are not equal, False otherwise
        :rtype: bool
        """

        return self.backend.__ne__(other)
    
    def __getitem__(self, key) -> any: # type: ignore
        
        """
        Random access operator for VCSC. 

        As of right now, this only supports random access of a single element.

        :param key: The index of the element to access
        :type key: int
        :return: The value of the element at the index
        :rtype: any
        """
        
        return self.backend.__getitem__(key)
    
    def getValues(self, outerIndex: int) -> list: 

        """
        Returns the unique values of a column or row depending on storage order.
        
        Note: Whether the values are from a column or row depends on order of the matrix.
              A matrix stored in column-major order will return the values of a column.
        
        :param outerIndex: The index of the column or row to get the values of
        :type outerIndex: int
        :return: A list containing the unique values of the column or row
        :rtype: list
        :raises IndexError: If the provided index is out of range
        """

        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.backend.getValues(outerIndex)
    
    def getIndices(self, outerIndex: int) -> list: 

        """
        Returns the indices of a column or row depending on storage order.

        Note: Whether the indices are from a column or row depends on order of the matrix.
                A matrix stored in column-major order will return the indices of a column.
        
        :param outerIndex: The index of the column or row to get the indices of
        :type outerIndex: int
        :return: A list containing the indices of the column or row
        :rtype: list
        :raises IndexError: If the provided index is out of range
        """

        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.backend.getIndices(outerIndex)
    
    def getCounts(self, outerIndex: int) -> list:

        """
        Returns the number of non-zero elements in a column or row depending on storage order.

        For example, if the matrix is:
            [1] 
            [1]
            [2]
        Then the list [1, 2] will be returned

        Note: Whether the counts are from a column or row depends on order of the matrix.
                A matrix stored in column-major order will return the counts of a column.

        :param outerIndex: The index of the column or row to get the counts of
        :type outerIndex: int
        :return: A list containing the counts of the column or row
        :rtype: list[Union[np.uint8, np.uint16, np.uint32, np.uint64]]
        :raises IndexError: If the provided index is out of range
        """

        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.backend.getCounts(outerIndex)
    
    def getNumIndices(self, outerIndex: int) -> list: 
        
        """
        Returns the number of unique values in a column or row depending on storage order.

        Note: Whether the number of indices are from a column or row depends on order of the matrix.
                A matrix stored in column-major order will return the number of indices of a column.
        
        :param outerIndex: The index of the column or row to get the number of indices of
        :type outerIndex: int
        :return: A list containing the number of indices of each column or row
        :rtype: list
        :raises IndexError: If the provided index is out of range
        """
        
        if outerIndex < 0:
            outerIndex += int(self.outerSize)
        elif outerIndex >= self.outerSize or outerIndex < 0: #type: ignore
            message = "Outer index out of range. Input: " + str(outerIndex) + " Range: [" + str(int(-self.outerSize) + 1) + ", " + str(int(self.outerSize) - 1) + "]"
            raise IndexError(message)
        return self.backend.getNumIndices(outerIndex)
    
    def append(self, matrix) -> None: 

        """
        Appends a matrix to the current matrix

        The appended matrix must be of the same type or a scipy.sparse.csc_matrix/csr_matrix 
        depending on the storage order of the current matrix. For a column-major matrix,
        the appended matrix will be appended to the end of the columns. For a row-major matrix,
        the appended matrix will be appended to the end of the rows.

        :param matrix: The matrix to append
        :type matrix: Union[VCSC, sp.sparse.csc_matrix, sp.sparse.csr_matrix]
        :raises TypeError: If the input matrix is not a supported type of matrix
        """

        if isinstance(matrix, VCSC) and self.order == matrix.order:
            self.backend.append(matrix.backend)
            self.rows += matrix.shape()[0] # type: ignore
            self.cols += matrix.shape()[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csc_matrix) and self.order == "Col":
            self.backend.append(matrix)
            self.rows += matrix.shape[0] # type: ignore
            self.cols += matrix.shape[1] # type: ignore
        elif isinstance(matrix, sp.sparse.csr_matrix) and self.order == "Row":
            self.backend.append(matrix.tocsc())
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



    def slice(self, start, end) -> VCSC:  # TODO fix for row major. Only broken in python

        """
        Returns a slice of the matrix.

        Currently, only slicing by storage order is supported. For example, if the matrix is stored in column-major order,
        Then the returned matrix will be a slice of the columns.
        
        :param start: The start index of the slice
        :type start: int
        :param end: The end index of the slice
        :type end: int
        :return: The slice of the matrix
        :rtype: VCSC
        """

        result = self
        result.backend = self.backend.slice(start, end)
        result.nnz = result.backend.nonZeros()

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
        Writes the matrix to a file. If the file name doesn't include .vcsc, it will be appended.

        :param filename: The name of the file to write to
        :type filename: str
        """

        self.backend.write(filename)

    def read(self, filename: str):

        """
        Function to read a VCSC formatted matrix from a file.
        This function should automatically determine the template type of the matrix.
        
        This can also read a .npz file, but it must be of a CSC, CSR, or COO format.

        :param filename: The name of the file to read from
        :type filename: str
        """

        if filename[-4:] == ".npz":
            self._npzConstruct(filename)
            return
        
        assert filename[-5:] == ".vcsc", "File must have a .vcsc extension"

        try:
            matFile = open(filename, "rb")
        except:
            raise IOError(f"Could not open file: {filename}. Check that the file is in the correct format or written from VCSC.write().")

        matFile.seek(16)
        valueByte = matFile.read(4)
        matFile.seek(20)
        indexSize = int(matFile.read(1)[0])
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
                raise ValueError(f"Invalid floating point flag byte in VCSC file: {filename}. Value: {valueByte}")
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
                raise ValueError(f"Invalid type size in VCSC file: {filename}. Value: {typeSize}")

        if indexSize == 1:
            self.indexType = np.uint8
        elif indexSize == 2:
            self.indexType = np.uint16
        elif indexSize == 4:
            self.indexType = np.uint32
        elif indexSize == 8:
            self.indexType = np.uint64
        else:
            raise ValueError("Invalid index size")
        
        if isColumnMajor:
            self.order = "Col"
        else:
            self.order = "Row"

        # no try-catch is used because of the error handling in the C++ code becomes obfuscated by python.
        self.backend = eval(str("PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)))(filename)
        # try:
        # self.backend.read(filename)
        # except:
            # raise IOError("Could not open file: " + filename + ". Check that the file is in the correct format or written from VCSC.write().")

        self.rows = self.backend.rows
        self.cols = self.backend.cols
        self.nnz = self.backend.nonZeros()
        self.innerSize = self.backend.innerSize
        self.outerSize = self.backend.outerSize
        self.bytes = self.backend.byteSize
    
    
    def _npzConstruct(self, moduleName: str, secondary: str = "csc"): 

        """
        This is a wrapper function to construct a VCSC matrix from a .npz file. This still creates a scipy.sparse matrix, but
        allows for a convenient way to read the file.

        :param moduleName: The npz file name
        :type moduleName: str
        :param secondary: The secondary format to convert the matrix to a valid form if the npz file is not one with an available constructor. i.e. BSR -> CSC. The default is "csc"
        :type secondary: str
        :raises TypeError: If the secondary format is not a valid format.
        """

        tempMat = sp.sparse.load_npz(moduleName)
        self.dtype = tempMat.dtype
        secondary = secondary.lower()

        # Secondary format is used to convert the input matrix to a different format
        # If a BSR matrix is writtent to the npz file, then the secondary format can be 
        # used to convert the matrix to a valid format.
        if tempMat.format.lower() in ("bsr", "dia", "lil", "dok"):
            if secondary == "csc":
                tempMat = tempMat.tocsc()
            elif secondary == "csr":
                tempMat = tempMat.tocsr()
            elif secondary == "coo":
                tempMat = tempMat.tocoo()
            else:
                raise TypeError("Invalid secondary format. Use CSC, CSR, or COO.")

        if(tempMat.format == "csc"):
            self.order = "Col"
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)
            self._CSconstruct(moduleName, tempMat)

        elif(tempMat.format == "csr"):
            self.order = "Row"
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)
            self._CSconstruct(moduleName, tempMat)
        elif(tempMat.format == "coo"):
            moduleName = "PyVSparse._PyVSparse._VCSC._" + str(self.dtype) + "_" + str(np.dtype(self.indexType)) + "_" + str(self.order)
            self._COOconstruct(moduleName, tempMat)
        else:
            raise TypeError("Input matrix does not have a valid format! Use CSC, CSR, or COO.")
        

        
    def _CSconstruct(self, moduleName: str, spmat):
        """
        Private helper function to construct a VCSC matrix from a scipy.sparse CSC or CSR matrix.
        This uses the Eigen::SparseMatrix<T> constructor in C++. Pybind11 handles the conversion.

        A pure Python implementation of a CSR/C matrix could be used to make a VCSC matrix, but 
        that is not implemented at this time. The C++ backend *should* support it though.

        :param moduleName: The name of the module to construct the VCSC matrix from
        :type moduleName: str
        :param spmat: The input matrix
        :type spmat: Union[sp.sparse.csc_matrix, sp.sparse.csr_matrix]
        """

        self.rows: np.uint32 = spmat.shape[0]
        self.cols: np.uint32 = spmat.shape[1]
        self.nnz = spmat.nnz

        if(self.order == "Col"):
            self.innerSize: np.uint32 = self.rows
            self.outerSize: np.uint32 = self.cols
        else:
            self.innerSize: np.uint32 = self.cols
            self.outerSize: np.uint32 = self.rows

        self.backend = eval(str(moduleName))(spmat)
        self.bytes: np.uint64 = self.backend.byteSize

    def _COOconstruct(self, moduleName: str, spmat): 

        """
        Private helper function to construct a VCSC matrix from a scipy.sparse COO matrix.
        In C++, the constructor expects std::tuple<indexT, indexT, T> for each non-zero element.

        C++ declaration:
        template <typename T2, typename indexT2>
        VCSC(std::vector<std::tuple<indexT2, indexT2, T2>>& entries, uint64_t num_rows, uint32_t num_cols, uint32_t nnz);

        :param moduleName: The name of the module to construct the VCSC matrix from
        :type moduleName: str
        :param spmat: The input matrix
        :type spmat: sp.sparse.coo_matrix
        """

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

        self.backend = eval(str(moduleName))(coords, self.rows, self.cols, spmat.nnz)
        self.bytes: np.uint64 = self.backend.byteSize
