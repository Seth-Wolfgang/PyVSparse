# PyVSparse

Python wrapper for [IVSparse](https://github.com/Seth-Wolfgang/IVSparse) \
Link to paper on [ArXiv](https://arxiv.org/abs/2309.04355)

## Documentation

Link to GitHub Pages: [Docs](https://seth-wolfgang.github.io/PyVSparse/)

IVSparse is a library for Index-and-Value Compressed Sparse Column (IVCSC) and Value Compressed Sparse Column (VCSC). \
Each are methods to losslessly compress redundant sparse matrices while keeping them usable (iterable).

For a vector:

```text
[1]
[2]
[1]
[3]
[1]
```

## VCSC

VCSC will be formatted as:

```text
Value = [1 2 3]
Index = [0 2 4 1 3]
Count = [3 1 1]
```

Where the indices of the `1` are:

- the first 3 values in `Index` (0, 2, and 4)
- first values of `Value` (1)
- and `count` (3).

This process is repeated for each vector, so a `1` in another vector will have its own "run".

For the `2` and `3`, there is only 1 index of each, and they are listed in order with the `2` being at index `4` and `3` at index `3`.

VCSC is typically faster for smaller matrices than IVCSC. This compression format will do no worse than COO, but can be worse than CSR/CSC if the data is not redundant.

## IVCSC

For the same vector:

```text
[1]
[2]
[1]
[3]
[1]
```

IVCSC will format it as

```text
[1] [1] [0 2 2] [0] [2] [1] [4] [0] [3] [1] [3] [0]
 V   W   I       D   V   W   I   D   V   W   I   D
```

Key:

- V - Value
- W - Width
- I - indices
- D - Delimiter

IVCSC uses bytepacking and positive delta encoding to achieve further compression, typically, at the cost of some performance. For the first set of indices, we see the string [0 2 2]. Each index is a delta, so the sum of the previous ones is the index's value, 0 + 2 + 2 = 4, so the index is 4. `W` is the byte width of the indices. A width of 1 means each index only takes 1 byte to store, and this is dynamically set by the compression algorithm. The width is the smallest number of bytes to store the largest index delta, so an the indices [1 1,000,000] will be stored in 3 byte deltas. IVCSC works best if each unique value is close together in a vector.

### Dependancies

- numpy
- scipy
- matplotlib
- python 3.9 or higher

Install by:

```bash
pip install PyVSparse
```

Which should also downlaod the dependencies.
Or use this repo by:

```bash
git clone https://github.com/Seth-Wolfgang/PyVSparse.git
cd PyVSparse
git submodule update --init --recursive
cd ..
pip install ./PyVSparse
```

However, building will require [pybind11](https://github.com/pybind/pybind11) and [scikit-build-core](https://github.com/scikit-build/scikit-build-core)

## Sample Program

```python
from PyVSparse.vcsc import VCSC
from PyVSparse.ivcsc import IVCSC
import scipy as sp
import numpy as np

# Also works for CSR!
CSC_Mat = sp.sparse.random(5, 5, format='csc', dtype = np.int8, density=1)

# Only SpMM or SpMV works for now
Dense_Vec = np.ones((5, 1), dtype = np.int8)

# Convert from CSC
VCSC_Mat = VCSC(CSC_Mat) # Will soon support VCSC_Mat = VCSC(CSC_Mat, indexType = np.int8)
IVCSC_Mat = IVCSC(CSC_Mat, major = "row") # the storage order can be set to "col" or "row"

# SpMV (will return np.ndarray)
IVCSC_Result = VCSC_Mat * Dense_Vec
VCSC_Result = IVCSC_Mat * Dense_Vec
CSC_Result = CSC_Mat * Dense_Vec

# Output
print("CSC: \n", CSC_Result)
print("VCSC: \n", VCSC_Result)
print("IVCSC: \n", IVCSC_Result) 
```

Returns

```bash
CSC: 
 [[-38]
 [-99]
 [ 14]
 [ 22]
 [ 81]]
VCSC: 
 [[-38]
 [-99]
 [ 14]
 [ 22]
 [ 81]]
IVCSC: 
 [[-38]
 [-99]
 [ 14]
 [ 22]]
```

## Todo

1. parallelize compilation
2. Compatability for windows and mac

## To cite IVSparse

```text
@misc{ivsparse,
  doi = {10.48550/ARXIV.2309.04355},
  url = {https://arxiv.org/abs/2309.04355},
  author = {Ruiter,  Skyler and Wolfgang,  Seth and Tunnell,  Marc and Triche,  Timothy and Carrier,  Erin and DeBruine,  Zachary},
  keywords = {Data Structures and Algorithms (cs.DS),  Machine Learning (cs.LG),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Value-Compressed Sparse Column (VCSC): Sparse Matrix Storage for Redundant Data},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
