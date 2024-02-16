Operator Usage
==============

Example source code can be found in ./examples/operators.py.

..  code-block:: python
    
    import PyVSparse
    import numpy as np
    import scipy as sp

    csc_mat = sp.sparse.random(3, 3, format='csc', dtype=np.float32, density=1)
    
    vcsc = PyVSparse.VCSC(csc_mat)
    ivcsc = PyVSparse.IVCSC(csc_mat)

.. note:: Both formats use the same syntax for operations


Scalar Multiplication
---------------------

..  code-block:: python
   
    vcsc *= 2 
    # or 
    vcsc = vcsc * 2 # But this is more expensive because it creates a new matrix

    ivcsc *= 2
    # or
    ivcsc = ivcsc * 2

Vector Multiplication
---------------------

Because Sparse * Dense (SpMV) operations produce dense objects, the result will be a numpy array.
Sparse * Sparse operations are not supported.

..  code-block:: python
    
    vector = np.array([1, 2, 3], dtype=np.float32)

    dense_numpy_array1 = vcsc * vector
    dense_numpy_array2 = ivcsc * vector


Matrix Multiplication
---------------------


Like vector multiplication, Sparse * Sparse operations are not supported.
SpMM is supported and returns a dense numpy array.

.. note:: The `*` operator is used for SpMM, unlike the `*` operator in scipy (which uses `@`).

..  code-block:: python
    
    dense_numpy_array3 = vcsc * csc_mat.toarray() # toarray() creates a dense numpy array
    dense_numpy_array4 = ivcsc * csc_mat.toarray()

Transpose
---------

Transpose is supported for both VCSC and IVCSC matrices.
Unfortunately, transpose is very slow becasue of how the data is stored.

..  code-block:: python
    
    vcsc_transpose = vcsc.transpose()
    # or an in place transpose
    vcsc.transpose(inplace=True)

    ivcsc_transpose = ivcsc.transpose()
    # or an in place transpose
    ivcsc.transpose(inplace=True)


Equality
--------

..  code-block:: python
    
    print(vcsc == ivcsc) # NotImplementedError
    print(vcsc == vcsc) # True
    print(ivcsc == ivcsc) # True


Inequality
----------

..  code-block:: python
    
    print(vcsc != ivcsc) # NotImplementedError
    print(vcsc != vcsc) # False
    print(ivcsc != ivcsc) # False

Random Access
-------------

..  code-block:: python
    
    print(vcsc[0, 0]) # Get the value at row 0, column 0
    print(ivcsc[0, 0]) # Get the value at row 0, column 0

.. note:: There is no way to write a value to this location.


Max
---

..  code-block:: python
    
    print(vcsc.max()) 
    print(vcsc.max(axis=0)) # Max of each column
    print(vcsc.max(axis=1)) # Max of each row
    
    print(ivcsc.max())
    print(ivcsc.max(axis=0)) # Max of each column
    print(ivcsc.max(axis=1)) # Max of each row

Min
---

..  code-block:: python
    
    print(vcsc.min()) 
    print(vcsc.min(axis=0)) # Min of each column
    print(vcsc.min(axis=1)) # Min of each row
    
    print(ivcsc.min())
    print(ivcsc.min(axis=0)) # Min of each column
    print(ivcsc.min(axis=1)) # Min of each row

.. Note:: Because of the way the matrix is stored, minimums that are zero are very expensive to compute. The returned value is the minimum non-zero value unless all values are zero or nnz < rows*cols (for min(None))

Sum
---

..  code-block:: python 
    
    print(vcsc.sum())
    print(vcsc.sum(axis=0)) # Sum of each column
    print(vcsc.sum(axis=1)) # Sum of each row
    
    print(ivcsc.sum())
    print(ivcsc.sum(axis=0)) # Sum of each column
    print(ivcsc.sum(axis=1)) # Sum of each row

Trace
------

..  code-block:: python
    
    print(vcsc.trace())
    print(ivcsc.trace())
