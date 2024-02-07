---
title: Sparse Matrices
description: How to store and solve problems with many zeros
sort: 10
---
# Sparse Matrices

* * *

## Definition

A Sparse matrix is a matrix with few non-zero entries.

A \\(m \times n\\) matrix is called sparse if it has <span>\\(O(min(m, n))\\)</span> non-zero entries. 

## Goals

A sparse matrix aims to store large matrices efficiently without storing many zeros, which allows for more economical computations.

The number of operations required to add two dense matrices <span>\\(\mathbf{P}\\)</span>, <span>\\(\mathbf{Q}\\)</span> is  <span>\\(O(n(\mathbf{P}) \times nnz(\mathbf{ }))\\)</span> where <span>\\n(\mathbf{X})\\)</span> is the number of elements in <span>\\(\mathbf{X}\\)</span>.

The number of operations required to add two sparse matrices <span>\\(\mathbf{P}\\)</span>, <span>\\(\mathbf{Q}\\)</span> is  <span>\\(O(nnz(\mathbf{P}) \times nnz(\mathbf{Q}))\\)</span> where <span>\\nnz(\mathbf{X})\\)</span> is the number of non-zero elements in <span>\\(\mathbf{X}\\)</span>.

## Storage Solutions

Let's explore ways to store the following example:
<div>\[\mathbf{A}= \begin{bmatrix} 1.0 & 0 & 0 & 2.0 & 0 \\ 3.0 & 4.0 & 0 & 5.0 & 0 \\ 6.0 & 0 & 7.0 & 8.0 & 9.0 \\ 0 & 0 & 10.0 & 11.0 & 0 \\ 0 & 0 & 0 & 0 & 12.0 \end{bmatrix}.\]</div>


### Dense Matrices

A \\(m \times n\\) matrix is called dense if it stores all values including zeros. To store <span>\\(\mathbf{A}\\)</span> above:
<div>\[A = \begin{bmatrix} 1.0 & 0 & 0 & 2.0 & 0 \\ 3.0 & 4.0 & 0 & 5.0 & 0 \\ 6.0 & 0 & 7.0 & 8.0 & 9.0 \\ 0 & 0 & 10.0 & 11.0 & 0 \\ 0 & 0 & 0 & 0 & 12.0 \end{bmatrix}.\]</div>

To store the matrix, all components are saved in row-major order.  For <span>\\(\mathbf{A}\\)</span> given above, we would store:
<div>\[A_{dense} = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 & 5.0 & 6.0 & 7.0 & 8.0 & 9.0 \end{bmatrix}.\]</div>

The dimensions of the matrix are stored separately:
<div>\[A_{shape} = \\((nrow, ncol)\\)\]</div>

The benefits of this method include simplicity and easy-blocked format.

### Coordinate Format (COO)

 COO stores arrays of row indices, column indices and the corresponding non-zero data values in any order. This format provides fast methods to construct sparse matrices and convert to different sparse formats. For <span>\\({\bf A}\\)</span> the COO format is:

$$\textrm{data} = \begin{bmatrix} 12.0 & 9.0 & 7.0 & 5.0 & 1.0 & 2.0 & 11.0 & 3.0 & 6.0 & 4.0 & 8.0 & 10.0\end{bmatrix}$$

$$\textrm{row} = \begin{bmatrix} 4 & 2 & 2 & 1 & 0 & 0 & 3 & 1 & 2 & 1 & 2 & 3 \end{bmatrix}, \\ \textrm{col} = \begin{bmatrix} 4 & 4 & 2 & 3 & 0 & 3 & 3 & 0 & 0 & 1 & 3 & 2 \end{bmatrix} $$

(\textrm{row}\\) and \\(\textrm{col}\\) are arrays of integers.

\\(\textrm{data}\\) is an array of doubles.

How to interpret: The first entries of \\(\textrm{data}\\), \\(\textrm{row}\\), \\(\textrm{col}\\) are 12.0, 4, 4, respectively, meaning there is a 12.0 at position (4, 4) of the matrix; second entries are 9.0, 2, 4, so there is a 9.0 at (2, 4). 

This method does not store the zero elements.

### Compressed Sparce Row (CSR)

 CSR encodes rows offsets, column indices and the corresponding non-zero data values. The row offsets are defined by the followign recursive relationship (starting with \\(\textrm{rowptr}[0] = 0\\)):

<div>\[ \textrm{rowptr}[j] = \textrm{rowptr}[j-1] + \mathrm{nnz}(\textrm{row}_{j-1}), \\ \]</div>

where \\(\mathrm{nnz}(\textrm{row}_k)\\) is the number of non-zero elements in the <span>\\(k^{th}\\)</span> row. Note that the length of \\(\textrm{rowptr}\\) is <span>\\(n_{rows} + 1\\)</span>, where the last element in \\(\textrm{rowptr}\\) is the number of nonzeros in <span>\\(A\\)</span>. For <span>\\({\bf A}\\)</span> the CSR format is:

$$\textrm{data} = \begin{bmatrix} 1.0 & 2.0 & 3.0 & 4.0 & 5.0 & 6.0 & 7.0 & 8.0 & 9.0 & 10.0 & 11.0 & 12.0 \end{bmatrix}$$

$$\textrm{col} = \begin{bmatrix} 0 & 3 & 0 & 1 & 3 & 0 & 2 & 3 & 4 & 2 & 3 & 4\end{bmatrix}$$

$$\textrm{rowptr} = \begin{bmatrix} 0 & 2 & 5 & 9 & 11 & 12 \end{bmatrix}$$


(\textrm{rowptr}\\) contains the row offset (array of <span>\\(n + 1\\)</span>)  integers).

\\(\textrm{col}\\) contains column indices (array of <span>\\(nnz\\)</span>)  integers).

\\(\textrm{data}\\) contains non-zero elements (array of <span>\\(nnz\\)</span>)  doubles).

How to interpret: The first two entries of \\(\textrm{rowptr}\\) gives us the elements in the first row. Interval [0, 2) of \\(\textrm{data}\\) and \\(\textrm{col}\\), corresponding to two (data, column) pairs: (1.0, 0) and (2.0, 3), means the first row has 1.0 at column 0 and 2.0 at column 3. The second and third entries of \\(\textrm{rowptr}\\) tells us [2, 5) of \\(\textrm{data}\\) and \\(\textrm{col}\\) corresponds to the second row. The three pairs (3.0, 0), (4.0, 1), (5.0, 3) means in the second row, there is a 3.0 at column 0, a 4.0 at column 1, and a 5.0 at column 3. 

This method does not store the zero elements and provides fast arithmetic operations between sparse matrices, and fast matrix vector product. 

## CSR Matrix Vector Product Algorithm

The following code snippet performs CSR matrix vector product for square matrices:

```python
import numpy as np
def csr_mat_vec(A, x):
  Ax = np.zeros_like(x)
  for i in range(x.shape[0]):
    for k in range(A.rowptr[i], A.rowptr[i+1]):
      Ax[i] += A.data[k]*x[A.col[k]]
  return Ax
```

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-11-sparse.html)
## ChangeLog

* 2022-03-06 Victor Zhao [chenyan4@illinois.edu](mailto:chenyan4@illinois.edu): Added instructions on how to interpret COO and CSR
* 2020-03-01 Peter Sentz: extracted material from previous reference pages
