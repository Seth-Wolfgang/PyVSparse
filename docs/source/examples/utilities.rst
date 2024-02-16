Utilites
========

Example source code can be found in ./examples/utilites.py.


..  code-block:: python
    
    import PyVSparse
    import numpy as np
    import scipy as sp
    csc_mat = sp.sparse.random(4, 4, format='csc', dtype=np.int32, density=1)

    vcsc = PyVSparse.VCSC(csc_mat)
    ivcsc = PyVSparse.IVCSC(csc_mat)


Slice
-----
Slicing is supported for both VCSC and IVCSC matrices, but only along the major axis.
Slicing along the minor axis is not supported.

..  code-block:: python
    
    vcsc_half = vcsc.slice(0, 2)
    ivcsc_half = ivcsc.slice(0, 2)

Appending
---------

Appending is supported for both VCSC and IVCSC matrices, but only along the major axis.
CSC/CSR matrices can be appended to I/VCSC.

..  code-block:: python
    
    vcsc.append(csc_mat)
    ivcsc.append(csc_mat)
    ivcsc.append(ivcsc)

    # NOT ALLOWED
    vcsc.append(ivcsc)
    ivcsc.append(vcsc)

VCSC Indices, Values, and Counts
---------------------------------

VCSC matrices store their indices, values, and counts in separate arrays. These can be copied to a list

..  code-block:: python
    
    index = 0 # The major index to access i.e. column or row to get
    indices = vcsc.getIndices(index)
    values = vcsc.getValues(index)
    counts = vcsc.getCounts(index)
   
    # number of indices is also available
    numIndices = vcsc.getNumIndices(index)

.. note:: IVCSC does not support this becasue of how its stored.

Shape 
-----

..  code-block:: python
    
    print(vcsc.shape())
    print(ivcsc.shape()[1])


Bytesize
--------
PyVSparse will tell you the exact amount of memory your matrix is using.
This does not include class attributes, as your matrix should be using the 
overwhelming majority of memory.

..  code-block:: python
  
    print(vcsc.byteSize())
    print(ivcsc.byteSize())
