Constructor Usage
=================

Example source code can be found in ./examples/constructor.py.

..  code-block:: python
   :caption: Construction

    # Create a 3x3 sparse matrix. Any dtype works as long as its a C++ supported type
    csc_mat = sp.sparse.random(3, 3, format='csc', dtype=np.float32, density=1)
    csr_mat = sp.sparse.random(3, 3, format='csr', dtype=np.int32, density=1)
    coo_mat = sp.sparse.random(3, 3, format='coo', dtype=np.float64, density=1)
    # Note: There must be at least 1 nonzero value in the matrix.

    # Create a PyVSparse matrix from the CSC matrix
    vcsc_mat1 = PyVSparse.VCSC(csc_mat)
    ivcsc_mat1 = PyVSparse.IVCSC(csc_mat)

    # Create a PyVSparse matrix from the CSR matrix
    vcsr_mat2 = PyVSparse.VCSC(csr_mat)
    ivcsc_mat2 = PyVSparse.IVCSC(csr_mat)

    # Create a PyVSparse matrix from the COO matrix
    vcsc_mat3 = PyVSparse.VCSC(coo_mat)
    ivcsc_mat3 = PyVSparse.IVCSC(coo_mat)
    

These can be transformed back into SciPy CSC or CSR matrices

..  code-block:: python
    
    csc_mat1 = vcsc_mat1.tocsc()
    csr_mat1 = ivcsc_mat1.tocsr() # It does not matter how it was created

    print(csc_mat1 == csc_mat)


VCSC and Index Types
---------------------
VCSC will store row/column indicies as 4 byte unsigned integers by default
But this can be adjusted during construction.

# Create a VCSC matrix with 2 byte unsigned integers

..  code-block:: python
    
    vcsc_mat = PyVSparse.VCSC(csc_mat, index_type=np.uint16)

This is useful for reducing memory usage when the indices are small enough to fit in 2 bytes.
These are checked during construction to ensure that the indices are not too large for the specified type.
An error is thrown if the indices are too large.

By default, VCSC indices are 4 byte unsigned integers. In python, all indices are unsigned integers to reduce compile time.

.. note:: IVCSC matrices do not have this option, as the indices are compressed to their smallest size.

Major Axes
----------

VCSC and IVCSC matrices can be constructed with either row or column major axes.
This is useful for optimizing memory access patterns for different operations.

..  code-block:: python
    
    vcsc_mat5 = PyVSparse.VCSC(csc_mat, order="row") 
    ivcsc_mat5 = PyVSparse.IVCSC(csc_mat, order="row")

`order="row"` will store the matrix in row major order, while `order="col"` will store the matrix in column major order.
By default, the matrices are stored in column major order. Capitalization does not matter.

