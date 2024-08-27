---
title: LU Decomposition for Solving Linear Equations
description: Computing and using LU decomposition.
sort: 9
author:
  - CS 357 Course Staff
changelog:
  - 
    name: Kaiyao Ke
    netid: kaiyaok2
    date: 2024-02-13
    message: aligned notes with slides, added examples and refactored existing notes

  - 
    name: Yuxuan Chen
    netid: yuxuan19
    date: 2023-10-20
    message: update lu_decomp() code

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-02-28
    message: fix error in ludecomp() code

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-02-22
    message: update properties for solving using LUP

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-01-14
    message: removes demo links

  - 
    name: John Doherty
    netid: jjdoher2
    date: 2017-11-02
    message: fixed typo in back substitution

  - 
    name: Arun Lakshmanan
    netid: lakshma2
    date: 2017-11-02
    message: minor fix in lup_solve(), add changelog

  - 
    name: Nathan Bowman
    netid: nlbowma2
    date: 2017-10-25
    message: added review questions

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-10-23
    message: fix links

  - 
    name: Matthew West
    netid: mwest
    date: 2017-10-20
    message: minor fix in back_sub()

  - 
    name: Nathan Bowman
    netid: nlbowma2
    date: 2017-10-19
    message: minor existence of LUP

  - 
    name: Luke Olson
    netid: lukeo
    date: 2017-10-17
    message: update links

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-10-17
    message: fixes

  - 
    name: Matthew West
    netid: mwest
    date: 2017-10-16
    message: first draft complete

---

## Learning Objectives
- Understand the use of linear system of equations.
- Use known data to set up linear system of equations for practical problems.
- Describe the factorization $${\bf A} = {\bf LU}$$.
- Compare the cost of LU with other operations such as matrix-matrix multiplication.
- Implement an LU decomposition algorithm.
- Given an LU decomposition for $${\bf A}$$, solve the system $${\bf Ax} = {\bf b}$$.
- Give examples of matrices for which pivoting is needed.
- Implement an LUP decomposition algorithm.
- Manually compute LU and LUP decompositions.
- Compute and use LU decompositions using library functions.

## Basic Idea: The “Undo” button for Linear Operations
Matrix-vector multiplication: given the data $${\bf x}$$ and the operator $${\bf A}$$, we can find $${\bf y}$$ such that $${\bf y = Ax}$$:

$$
\bf{x} \hspace{5mm} {\xRightarrow[transformation]{A}} \hspace{5mm} \bf{y}
$$

What if we know $${\bf y}$$ but not $${\bf x}$$? We need to “undo” the transformation:

$$
\bf{y} \hspace{5mm} {\xRightarrow[?]{A^{-1}}} \hspace{5mm} \bf{x}  \hspace{3cm} \textbf{Solve}\hspace{3mm} Ax=y \hspace{3mm}\text{for} \hspace{3mm}\bf{x}
$$

### Example: Undoing the Transformation

Suppose we have known operator $${\bf A}$$, known data $${\bf y}$$, and unkown data $${\bf x}$$ that satisfies the relationship $${\bf y = Ax}$$. The values of $${\bf A}$$ and $${\bf y}$$ are given below.

$$
\textbf{A} =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}\hspace{5mm} 
\textbf{x} =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} \hspace{5mm} \text{and} \hspace{5mm}
\textbf{y} =
\begin{bmatrix}
5 \\
11
\end{bmatrix}
.
$$

How can we solve for $$\textbf{x} = [x_1, x_2]^T$$?

<details>
    <summary><strong>Answer</strong></summary>

We construct the following set of linear equations:
$$
\begin{cases}
    x_1 + 2x_2 = 5\\
    3x_1 + 4x_2 = 11
\end{cases} \hspace{5mm}
$$
Then the solution is:
$$
\begin{cases}
    x_1 = 1\\
    x_2 = 2
\end{cases} \hspace{5mm} \text{, or} \hspace{5mm}
\textbf{x} =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
$$

</details>

### Example:  Image Blurring and Recovery
The original image displaying an SSN number is stored as a $$2D$$ array of real numbers between $$0$$ and $$1$$ ($$0$$ represents a white pixel, $$1$$ represents a black pixel).It has $$40$$ rows of pixels and $$100$$ columns of pixels. We can flatten the $$2D$$ array into a $$1D$$ array $$\bf{x}$$ containing the $$1D$$ data with dimension $$4000$$. We can then apply blurring operation to data $$\bf{x}$$, i.e.

$$
\bf{y} = \bf{A}\bf{x}
$$

where $$\bf{A}$$ is the blur operator and $$\bf{y}$$ is the blurred image:
<br/>

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/ssn_blur.png" width=800/> </div>

<br/>

To "undo" blurring to recover original image, we solve a linear system of equations using the blur operator $$\bf{A}$$ and the blurred image $$\bf{y}$$. The following graph shows the transformation when $$\bf{y}$$does not have any noise ("clean data"):
<br/>

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/ssn_undo_blur.png" width=800/> </div>

<br/>

It is also possible to recover $$\bf{x}$$ with a certain extent of noise:

<br/>

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/ssn_undo_blur_noise.png" width=800/> </div>

<br/>

To answer "How much noise can we add and still be able to recover meaningful information from the original image?" and "At which point this inverse transformation fails?", we need information about the sensitivity of the “undo” operation covered later in this course.

## Back Substitution Algorithm for Upper Triangular Systems

To actually solve $${\bf A x} = {\bf b}$$, we can start with an "easier" system of equations. Let’s consider triangular matrices - the **_back substitution algorithm_** solves the linear system $${\bf U x} = {\bf b}$$ where $${\bf U}$$ is an upper-triangular matrix. 

An upper-triangular linear system $${\bf U}{\bf x} = {\bf b}$$ can be written in matrix form:

$$
\begin{bmatrix}
U_{11} & U_{12} & \ldots & U_{1n} \\
0 & U_{22} & \ldots & U_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \ldots & U_{nn} \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_n
\end{bmatrix}.
$$

Observe that the upper-triangular system $${\bf U}x = b$$ can be written as the set of linear equations:

$$
\begin{matrix}
U_{11} x_1 & + & U_{12} x_2 & + & \ldots & + & U_{1n} x_n & = & b_1 \\
           &   & U_{22} x_2 & + & \ldots & + & U_{2n} x_n & = & b_2 \\
           &   &            &   & \ddots &   & \vdots     & = & \vdots \\
           &   &            &   &        &   & U_{nn} x_n & = & b_n.
\end{matrix}
$$

The back substitution solution works from the bottom up to give:

$$
\begin{aligned}
x_n &= \frac{b_n}{U_{nn}} \\
x_{n-1} &= \frac{b_{n-1} - U_{n-1,n} x_n}{U_{n-1,n-1}} \\
&\vdots \\
x_1 &= \frac{b_1 - \sum_{j=2}^n U_{1j} x_j}{U_{11}}.
\end{aligned}
$$

So the general form of solution is:

$$
x_n = \frac{b_n}{U_{nn}}; \hspace{1cm} x_i = \frac{b_i - \sum_{j=i+1}^n U_{ij} x_j}{U_{ii}} \hspace{5mm} \text{for i = n-1, n-2, ..., 1}
$$

Notice that there are $$n$$ divisions, $$\frac{n(n-1)}{2}$$ subtractions / additions, and $$\frac{n(n-1)}{2}$$ multiplications, hence the **computational complexity** is $$\bf{O(n^2)}$$.

Alternatively, we can also write $${\bf U}x = b$$ as a linear combination of the columns of $$\bf{U}$$:

$$
x_1 \hspace{1mm} \textbf{U}[:\hspace{1mm},1] + x_2 \hspace{1mm} \textbf{U}[:\hspace{1mm},2] + \ldots + x_{n} \hspace{1mm} \textbf{U}[:\hspace{1mm},n] = \textbf{b}
$$

The properties of the back substitution algorithm are:

1. If any of the diagonal elements $$U_{ii}$$ are zero then the system is singular and cannot be solved.
2. If all diagonal elements of $${\bf U}$$ are non-zero then the system has a unique solution.
3. The number of operations for the back substitution algorithm is $$O(n^2)$$ as $$n \to \infty$$.

The code for the back substitution algorithm to solve $${\bf U x} = {\bf b}$$ is:
```python
import numpy as np
def back_sub(U, b):
    """x = back_sub(U, b) is the solution to U x = b
       U must be an upper-triangular matrix
       b must be a vector of the same leading dimension as U
    """
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(i+1, n):
            tmp -= U[i,j] * x[j]
        x[i] = tmp / U[i,i]
    return x
```

### Example: Back Substitution for an Upper Triangular System

$$
\begin{bmatrix}
2 & 3 & 1 & 1 \\
0 & 2 & 2 & 3 \\
0 & 0 & 6 & 4 \\
0 & 0 & 0 & 2 \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}
=
\begin{bmatrix}
2 \\ 2 \\ 6 \\ 4
\end{bmatrix}.
$$

How can we solve for $$ x = [x_1, x_2, x_3, x_4]^T $$?

<details>
    <summary><strong>Answer</strong></summary>
$$
2x_4 = 4 \Rightarrow x_4 = \frac{4}{2} = 2
$$

$$
6x_3 + 4x_4 = 6 \Rightarrow x_3 = \frac{6 - 4(2)}{6} = -\frac{2}{3}
$$

$$
2x_2 + 2x_3 + 3x_4 = 2 \Rightarrow x_2 = \frac{2 - 2(-\frac{2}{3}) - 3(2)}{2} = -\frac{10}{3}
$$

$$
2x_1 + 3x_2 + x_3 + x_4 = 2 \Rightarrow x_1 = \frac{2 - 3(-\frac{10}{3}) + (-\frac{2}{3}) + 2}{2} = \frac{22}{3}
$$
</details>


## Forward Substitution Algorithm for Lower Triangular Systems

The **_forward substitution algorithm_** solves the linear system $${\bf Lx} = {\bf b}$$ where $${\bf L}$$ is a lower triangular matrix. It is the reversed version of back substitution.

A lower-triangular linear system $${\bf L}{\bf x} = {\bf b}$$ can be written in matrix form:

$$
\begin{bmatrix}
L_{11} & 0         & \ldots & 0 \\
L_{21} & L_{22} & \ldots & 0 \\
\vdots    & \vdots    & \ddots & 0 \\
L_{n1} & L_{n2} & \ldots & L_{nn} \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_n
\end{bmatrix}.
$$

This can also be written as the set of linear equations:

$$
\begin{matrix}
L_{11} x_1 &   &               &   &        &   &               & = & b_1 \\
L_{21} x_1 & + & L_{22} x_2 &   &        &   &               & = & b_2 \\
\vdots        & + & \vdots        & + & \ddots &   &               & = & \vdots \\
L_{n1} x_1 & + & L_{n2} x_2 & + & \ldots & + & L_{nn} x_n & = & b_n.
\end{matrix}
$$

The forward substitution algorithm solves a lower-triangular linear system by working from the top down and solving each variable in turn. In math this is:

$$
\begin{aligned}
x_1 &= \frac{b_1}{L_{11}} \\
x_2 &= \frac{b_2 - L_{21} x_1}{L_{22}} \\
&\vdots \\
x_n &= \frac{b_n - \sum_{j=1}^{n-1} L_{nj} x_j}{L_{nn}}.
\end{aligned}
$$

So the general form of solution is:

$$
x_1 = \frac{b_1}{L_{11}}; \hspace{1cm} x_i = \frac{b_i - \sum_{j=1}^{i-1} L_{ij} x_j}{L_{ii}} \hspace{5mm} \text{for i = 2, 3, ..., n}
$$

Notice that there are also $$n$$ divisions, $$\frac{n(n-1)}{2}$$ subtractions / additions, and $$\frac{n(n-1)}{2}$$ multiplications, hence the **computational complexity** is $$\bf{O(n^2)}$$.


The properties of the forward substitution algorithm are:

1. If any of the diagonal elements $$L_{ii}$$ are zero then the system is singular and cannot be solved.
2. If all diagonal elements of $${\bf L}$$ are non-zero then the system has a unique solution.
3. The number of operations for the forward substitution algorithm is $$O(n^2)$$ as $$n \to \infty$$.

The code for the forward substitution algorithm to solve $${\bf L x} = {\bf b}$$ is:

```python
import numpy as np
def forward_sub(L, b):
    """x = forward_sub(L, b) is the solution to L x = b
       L must be a lower-triangular matrix
       b must be a vector of the same leading dimension as L
    """
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        tmp = b[i]
        for j in range(i):
            tmp -= L[i,j] * x[j]
        x[i] = tmp / L[i,i]
    return x
```

### Example: Forward-substitution for a Lower Triangular System

$$
\begin{bmatrix}
2 & 0 & 0 & 0 \\
3 & 2 & 0 & 0 \\
1 & 2 & 6 & 0 \\
1 & 3 & 4 & 2 \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix}
=
\begin{bmatrix}
2 \\ 2 \\ 6 \\ 4
\end{bmatrix}.
$$

How can we solve for $$x = [x_1, x_2, x_3, x_4]^T$$?

<details>
    <summary><strong>Answer</strong></summary>

$$
2x_1 = 2 \Rightarrow x_1 = 1
$$

$$
3x_1 + 2x_2 = 2 \Rightarrow x_2 = \frac{2-3}{2} = -0.5
$$

$$
1x_1 + 2x_2 + 6x_3 = 6 \Rightarrow x_3 = \frac{6-1+1}{6} = 1
$$

$$
1x_1 + 3x_2 + 4x_3 + 2x_4 = 4 \Rightarrow x_4 = \frac{4-1+1.5-4}{2} = 0.25
$$

</details>

## LU Decomposition Definition

To solve $${\bf A x} = {\bf b}$$ when $$\bf{A}$$ is a non-triangular matrix, We can perform LU factorization: given an $$n \times n$$ matrix $$\bf{A}$$, the **_LU decomposition_** of a matrix $${\bf A}$$ is the pair of matrices $${\bf L}$$ and $${\bf U}$$ such that:

1. \\({\bf A} = {\bf LU}\\)
2. $${\bf L}$$ is a lower-triangular matrix with all diagonal entries equal to 1
3. $${\bf U}$$ is an upper-triangular matrix.

$$
\begin{bmatrix}
1 & 0 & \ldots & 0 \\
L_{21} & 1 & \ldots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
L_{n1} & L_{n2} & \ldots & 1 \\
\end{bmatrix}
\begin{bmatrix}
u_{11} & U_{12} & \ldots & U_{1n} \\
0 & U_{22} & \ldots & U_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \ldots & U_{nn} \\
\end{bmatrix}
=
\begin{bmatrix}
A_{11} & A_{12} & \ldots & A_{1n} \\
A_{21} & A_{22} & \ldots & A_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
A_{n1} & A_{n2} & \ldots & A_{nn} \\
\end{bmatrix}
$$

The properties of the LU decomposition are:

1. The LU decomposition may not exist for a matrix $${\bf A}$$.
2. If the LU decomposition exists then it is unique.
3. The LU decomposition provides an efficient means of solving linear equations.
4. The reason that $${\bf L}$$ has all diagonal entries set to 1 is that this means the LU decomposition is unique. This choice is somewhat arbitrary (we could have decided that $${\bf U}$$ must have 1 on the diagonal) but it is the standard choice.
5. We use the terms **_decomposition_** and **_factorization_** interchangeably to mean writing a matrix as a product of two or more other matrices, generally with some defined properties (such as lower/upper triangular).

### Example: LU Decomposition

Consider a $$3 \times 3$$ matrix
$$
A =
\begin{bmatrix}
1 & 2 & 2 \\
4 & 4 & 2 \\
4 & 6 & 4
\end{bmatrix}
.$$

The LU factorization is
$${\bf A} = {\bf LU} =
\begin{bmatrix}
1 & 0 & 0 \\
4 & 1 & 0 \\
4 & 0.5 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 2 \\
0 & -4 & -6 \\
0 & 0 & -1
\end{bmatrix}.$$




## Solving LU-Decomposed Linear Systems

Knowing the LU decomposition for a matrix $${\bf A}$$ allows us to solve the linear system $${\bf A x} = {\bf b}$$ using a combination of forward and back substitution. In equations this is:

$$
\begin{aligned}
{\bf A x} &= {\bf b} \\
{\bf L U x} &= {\bf b} \\
{\bf U x} &= {\bf L}^{-1} {\bf b} \\
{\bf x} &= {\bf U}^{-1} ({\bf L}^{-1} {\bf b}),
\end{aligned}
$$

where we first evaluate $${\bf L}^{-1} {\bf b}$$ using forward substitution and then evaluate $${\bf x} = {\bf U}^{-1} ({\bf L}^{-1} {\bf b})$$ using back substitution.

An equivalent way to write this is to introduce a new vector $${\bf y}$$ defined by $$\bf{y} = {\bf U x}$$. Assuming the LU factorization of matrix $$\bf{A}$$ is known, we can solve the general system

$$
\bf{LU \hspace{1mm} x = b}
$$

By solving two triangular systems:

$$
\bf{Ly=b} \hspace{5mm} {\xRightarrow{\text{Solve for } \bf{y}}} \hspace{5mm} \text{Forward substitution with complexity } O(n^2)
$$

$$
\bf{Ux=y} \hspace{5mm} {\xRightarrow{\text{Solve for } \bf{x}}} \hspace{5mm} \text{Back substitution with complexity } O(n^2)
$$


We have thus replaced $${\bf A x} = {\bf b}$$ with *two* linear systems: $${\bf L y} = {\bf b}$$ and $${\bf U x} = {\bf y}$$. These two linear systems can then be solved one after the other using forward and back substitution.

The number of operations for the LU solve algorithm is $$O(n^2)$$ as $$n \to \infty$$.

The **_LU solve algorithm_** code for solving the linear system $${\bf L U x} = {\bf b}$$ is:

```python
import numpy as np
def lu_solve(L, U, b):
    """x = lu_solve(L, U, b) is the solution to L U x = b
       L must be a lower-triangular matrix
       U must be an upper-triangular matrix of the same size as L
       b must be a vector of the same leading dimension as L
    """
    y = forward_sub(L, b)
    x = back_sub(U, y)
    return x
```


## The LU Decomposition Algorithm

Given a matrix $${\bf A}$$ there are many different algorithms to find the matrices $${\bf L}$$ and $${\bf U}$$ for the LU decomposition. Here we will use the **_recursive leading-row-column LU algorithm_**. Let's first consider the simple case when $${\bf A}$$ is a $$2\times 2$$ matrix:

<br/>

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/lu_2x2.png" width=500/> </div>

<br/>

Use a similar idea to write $${\bf A}$$ in block form when $${\bf A}$$ is an $$n\times n$$ matrix:

<br/>

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/lu_nxn_block.png" width=300/> </div>

<br/>


Then $${\bf A} = {\bf LU}$$ can be re-written as:

$$
\begin{aligned}
\begin{bmatrix}
a_{11} & \boldsymbol{a}_{12} \\
\boldsymbol{a}_{21} & {\bf A}_{22}
\end{bmatrix}
&=
\begin{bmatrix}
1 & \boldsymbol{0} \\
\boldsymbol{\ell}_{21} & {\bf L}_{22}
\end{bmatrix}
\begin{bmatrix}
u_{11} & \boldsymbol{u}_{12} \\
\boldsymbol{0} & {\bf U}_{22}
\end{bmatrix}
\\
&=
\begin{bmatrix}
u_{11} & \boldsymbol{u}_{12} \\
u_{11} \boldsymbol{\ell}_{21} & (\boldsymbol{\ell}_{21} \boldsymbol{u}_{12} + {\bf L}_{22} {\bf U}_{22})
\end{bmatrix}.
\end{aligned}
$$

In the above block form of the $$n \times n$$ matrix $${\bf A}$$, the entry $$a_{11}$$ is a scalar, $$\boldsymbol{a}_{12}$$ is a $$1 \times (n-1)$$ row vector, $$\boldsymbol{a}_{12}$$ is an $$(n-1) \times 1$$ column vector, and $${\bf A}_{22}$$ is an $$(n-1) \times (n-1)$$ matrix.

Comparing the left- and right-hand side entries of the above block matrix equation we see that:

$$
\begin{aligned}
a_{11} &= u_{11} \\
\boldsymbol{a}_{12} &= \boldsymbol{u}_{12} \\
\boldsymbol{a}_{21} &= u_{11} \boldsymbol{\ell}_{21} \\
A_{22} &= \boldsymbol{\ell}_{21} \boldsymbol{u}_{12} + {\bf L}_{22} {\bf U}_{22}.
\end{aligned}
$$

These four equations can be rearranged to solve for the components of the $${\bf L}$$ and $${\bf U}$$ matrices as:

$$
\begin{aligned}
u_{11} &= a_{11}\\
\boldsymbol{u}_{12} &= \boldsymbol{a}_{12} \\
\boldsymbol{\ell}_{21} &= \frac{1}{u_{11}} \boldsymbol{a}_{21} \\
{\bf L}_{22} {\bf U}_{22} &= \underbrace{ {\bf A}_{22} - \boldsymbol{a}_{21} (a_{11})^{-1} \boldsymbol{a}_{12}}_{\text{Schur complement } S_{22}}.
\end{aligned}
$$

Notice that:

1. The first row of $$\bf{U}$$ is the first row of $$\bf{A}$$.
2. The first column of $$\bf{L}$$ is $$\frac{\text{the first column of  }\textbf{A}}{u_{11}}$$.
3. $${\bf L}_{22} {\bf U}_{22}$$ needs another factorization.

In other words, the first three equations above can be immediately evaluated to give the first row and column of $${\bf L}$$ and $${\bf U}$$. The last equation can then have its right-hand-side evaluated, which gives the **_Schur complement_** $$S_{22}$$ of $${\bf A}$$. We thus have the equation $${\bf L}_{22} {\bf U}_{22} = {\bf S}_{22}$$, which is an $$(n-1) \times (n-1)$$ LU decomposition problem which we can recursively solve.

The code for the **_recursive leading-row-column LU algorithm_** to find $${\bf L}$$ and $${\bf U}$$ for $${\bf A} = {\bf LU}$$ is:

```python
import numpy as np
def lu_decomp(A):
    """(L, U) = lu_decomp(A) is the LU decomposition A = L U
       A is any matrix
       L will be a lower-triangular matrix with 1 on the diagonal, the same shape as A
       U will be an upper-triangular matrix, the same shape as A
    """
    n = A.shape[0]
    if n == 1:
        L = np.array([[1]])
        U = A.copy()
        return (L, U)

    A11 = A[0,0]
    A12 = A[0,1:]
    A21 = A[1:,0]
    A22 = A[1:,1:]

    L11 = 1
    U11 = A11

    L12 = np.zeros(n-1)
    U12 = A12.copy()

    L21 = A21.copy() / U11
    U21 = np.zeros(n-1)

    S22 = A22 - np.outer(L21, U12)
    (L22, U22) = lu_decomp(S22)

    L = np.zeros(A.shape)
    L[0, 0] = L11
    L[0, 1:] = L12
    L[1:, 0] = L21
    L[1:, 1:] = L22
    U = np.zeros(A.shape)
    U[0, 0] = U11
    U[0, 1:] = U12
    U[1:, 0] = U21
    U[1:, 1:] = U22

    return (L, U)
```

Number of divisions: $$(n-1)+(n-2)+\ldots+1=\frac{n(n-1)}{2}$$

Number of multiplications: $$(n-1)^2+(n-2)^2+\ldots+1^2=\frac{n^3}{3} - \frac{n^2}{2} + \frac{n}{6}$$

Number of subtractions: $$(n-1)^2+(n-2)^2+\ldots+1^2=\frac{n^3}{3} - \frac{n^2}{2} + \frac{n}{6}$$

Hence the number of operations for the recursive leading-row-column LU decomposition algorithm is $$\bf{O(n^3)}$$ as $$n \to \infty$$.

### Example: LU decomposition

Consider the matrix
$$
\bf{A} =
\begin{bmatrix}
2 & 8 & 4 & 1 \\
1 & 2 & 3 & 3 \\
1 & 2 & 6 & 2 \\
1 & 3 & 4 & 2
\end{bmatrix}
.$$

How can we find an LU Decomposition for this matrix?
<details>
    <summary><strong>Answer</strong></summary>

Note that we use \(\bf {M}\) to keep track of the matrix (e.g. \(\bf {L_{22}U_{22}}\) in the first step) that needs recursive factorization.<br>

The first row of \(\bf{U}\) is the first row of \(\bf{A}\). <br>

The first column of \(\bf{L}\) is \(\frac{\text{the first column of  }\textbf{A}}{u_{11}}\). <br>

Also, as \({\bf L}_{22} {\bf U}_{22} =  {\bf A}_{22} - \boldsymbol{a}_{21} (a_{11})^{-1} \boldsymbol{a}_{12}\), we have the following after the first step (notice that we use the tensor product operator "\(\otimes\)" to denote the outer product of two vectors):

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
\textbf{0.5} & 1 & 0 & 0 \\
\textbf{0.5} & ? & 1 & 0 \\
\textbf{0.5} & ? & ? & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
2 & \textbf{8} & \textbf{4} & \textbf{1} \\
0 & ? & ? & ? \\
0 & 0 & ? & ? \\
0 & 0 & 0 & ?
\end{bmatrix}; \hspace{5mm}
\textbf{M} = 
\begin{bmatrix}
2 & 3 & 3 \\
2 & 6 & 2 \\
3 & 4 & 2 
\end{bmatrix} - 
\begin{bmatrix}
0.5 \\
0.5 \\
0.5
\end{bmatrix} \otimes
\begin{bmatrix}
8 \\
4 \\
1
\end{bmatrix} =
\begin{bmatrix}
-2 & 1 & 2.5 \\
-2 & 4 & 1.5 \\
-1 & 2 & 1.5 
\end{bmatrix}
$$

Similarly, in the second (recursive) step:

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0.5 & 1 & 0 & 0 \\
0.5 & \textbf{1} & 1 & 0 \\
0.5 & \textbf{0.5} & ? & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
2 & 8 & 4 & 1 \\
0 & -2 & \textbf{1} & \textbf{2.5} \\
0 & 0 & ? & ? \\
0 & 0 & 0 & ?
\end{bmatrix}; \hspace{5mm}
\textbf{M} = 
\begin{bmatrix}
4 & 1.5 \\
2 & 1.5 
\end{bmatrix} - 
\begin{bmatrix}
1 \\
0.5
\end{bmatrix} \otimes
\begin{bmatrix}
1 \\
2.5
\end{bmatrix} =
\begin{bmatrix}
3 & -1 \\
1.5 & 0.25
\end{bmatrix}
$$

The next step is:

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0.5 & 1 & 0 & 0 \\
0.5 & 1 & 1 & 0 \\
0.5 & 0.5 & \textbf{0.5} & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
2 & 8 & 4 & 1 \\
0 & -2 & 1 & 2.5 \\
0 & 0 & 3 & \textbf{-1} \\
0 & 0 & 0 & ?
\end{bmatrix}; \hspace{5mm}
\textbf{M} = 
\begin{bmatrix}
0.25 
\end{bmatrix} - 
\begin{bmatrix}
0.5
\end{bmatrix} \otimes
\begin{bmatrix}
-1
\end{bmatrix} =
\begin{bmatrix}
0.75
\end{bmatrix}
$$

So the final result is:

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0.5 & 1 & 0 & 0 \\
0.5 & 1 & 1 & 0 \\
0.5 & 0.5 & 0.5 & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
2 & 8 & 4 & 1 \\
0 & -2 & 1 & 2.5 \\
0 & 0 & 3 & -1 \\
0 & 0 & 0 & 0.75
\end{bmatrix}
$$
</details>

### Example: Matrix for which LU Decomposition Fails

An example of a matrix which has no LU decomposition is

$$
{\bf A} =
\begin{bmatrix}
2 & 8 & 4 & 1 \\
1 & 4 & 3 & 3 \\
1 & 2 & 6 & 2 \\
1 & 3 & 4 & 2
\end{bmatrix}
$$

Why?

<details>
    <summary><strong>Answer</strong></summary>

The first step of LU Decomposition is (notice that we use the tensor product operator "\(\otimes\)" to denote the outer product of two vectors):

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
\textbf{0.5} & 1 & 0 & 0 \\
\textbf{0.5} & ? & 1 & 0 \\
\textbf{0.5} & ? & ? & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
2 & 8 & 4 & 1 \\
0 & ? & ? & ? \\
0 & 0 & ? & ? \\
0 & 0 & 0 & ?
\end{bmatrix}; \hspace{5mm}
\bf{L_{22}U_{22}} = 
\begin{bmatrix}
4 & 3 & 3 \\
2 & 6 & 2 \\
3 & 4 & 2 
\end{bmatrix} - 
\begin{bmatrix}
0.5 \\
0.5 \\
0.5
\end{bmatrix} \otimes
\begin{bmatrix}
8 \\
4 \\
1
\end{bmatrix} =
\begin{bmatrix}
\textbf{0} & 1 & 2.5 \\
-2 & 4 & 1.5 \\
-1 & 2 & 1.5 
\end{bmatrix}
$$

The next update for the lower triangular matrix will result in a division by zero. LU factorization fails, so \(\bf A\) does not have a normal LU decomposition.
</details>

## Solving Linear Systems Using LU Decomposition

We can put the above sections together to produce an algorithm for solving the system $${\bf A x} = {\bf b}$$, where we first compute the LU decomposition of $${\bf A}$$ and then use forward and back substitution to solve for $${\bf x}$$.

The properties of this algorithm are:

1. The algorithm may fail, even if $${\bf A}$$ is invertible.
2. The number of operations in the algorithm is $$\mathcal{O}(n^3)$$ as $$n \to \infty$$.

The code for the **_linear solver using LU decomposition_** is:
```python
import numpy as np
def linear_solve_without_pivoting(A, b):
    """x = linear_solve_without_pivoting(A, b) is the solution to A x = b (computed without pivoting)
       A is any matrix
       b is a vector of the same leading dimension as A
       x will be a vector of the same leading dimension as A
    """
    (L, U) = lu_decomp(A)
    x = lu_solve(L, U, b)
    return x
```

## Pivoting

The LU decomposition can fail when the top-left entry in the matrix $${\bf A}$$ is zero or very small compared to other entries. **_Pivoting_** is a strategy to mitigate this problem by rearranging the rows and/or columns of $${\bf A}$$ to put a larger element in the top-left position.

There are many different pivoting algorithms. The most common of these are **_full pivoting_**, **_partial pivoting_**, and **_scaled partial pivoting_**. We will only discuss **_partial pivoting_** in detail.

1) **_Partial pivoting_** only rearranges the rows of $${\bf A}$$ and leaves the columns fixed.

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/partialPivoting.png" width=500/> </div>

<br/>
2) **_Full pivoting_** rearranges both rows and columns.

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/fullPivoting.png" width=500/> </div>

<br/>
3) **_Scaled partial pivoting_** approximates full pivoting without actually rearranging columns.


## LU Decomposition with Partial Pivoting

The **_LU decomposition with partial pivoting (LUP)_** of an $$n \times n$$ matrix $${\bf A}$$ is
the triple of matrices $${\bf L}$$, $${\bf U}$$, and $${\bf P}$$ such that:

1. \\({\bf P A} = {\bf LU} \\)
3. $${\bf L}$$ is an $$n \times n$$ lower-triangular matrix with all diagonal entries equal to 1.
4. $${\bf U}$$ is an $$n \times n$$ upper-triangular matrix.
2. $${\bf P}$$ is an $$n \times n$$ permutation matrix.

The properties of the LUP decomposition are:

1. The permutation matrix $${\bf P}$$ acts to permute the rows of $${\bf A}$$. This attempts to put large entries in the top-left position of $${\bf A}$$ and each sub-matrix in the recursion, to avoid needing to divide by a small or zero element.
2. The LUP decomposition always exists for a matrix $${\bf A}$$.
3. The LUP decomposition of a matrix $${\bf A}$$ is not unique.
4. The LUP decomposition provides a more robust method of solving linear systems than LU decomposition without pivoting, and it is approximately the same cost.

LU factorization with partial pivoting can be completed for any matrix A: Suppose you are at stage k and there is no non-zero entry on or below the diagonal in column k. At this point, there is nothing else you can do, so the algorithm leaves a zero in the diagonal entry of U. Note that the matrix U is singular, and so is the matrix A. Subsequent back substitutions using U will fail, but the LU factorization itself is still completed.

## Solving LUP decomposition linear systems

Knowing the LUP decomposition for a matrix $${\bf A}$$ allows us to solve the linear system $${\bf A x} = {\bf b}$$
by first applying $${\bf P}$$ and then using the LU solver.
In equations we start by taking $${\bf A x} = {\bf b}$$ and multiplying both sides by $${\bf P}$$, giving

$$
\begin{aligned}
{\bf Ax} &= {\bf b} \\
{\bf PAx} &= {\bf Pb} \\
{\bf LUx} &= {\bf Pb}.
\end{aligned}
$$

The code for the **_LUP solve algorithm_** to solve the linear system $${\bf L U x} = {\bf P b}$$ is:

```python
import numpy as np
def lup_solve(L, U, P, b):
    """x = lup_solve(L, U, P, b) is the solution to L U x = P b
       L must be a lower-triangular matrix
       U must be an upper-triangular matrix of the same shape as L
       P must be a permutation matrix of the same shape as L
       b must be a vector of the same leading dimension as L
    """
    z = np.dot(P, b)
    x = lu_solve(L, U, z)
    return x
```

The number of operations for the LUP solve algorithm is $$\mathcal{O}(n^2)$$ as $$n \to \infty$$.

## The LUP decomposition algorithm

Just as there are different LU decomposition algorithms, there are also different algorithms to find a LUP decomposition. Here we use the **_recursive leading-row-column LUP algorithm_**.

This algorithm is a recursive method for finding $${\bf L}$$, $${\bf U}$$, and $${\bf P}$$ so that $${\bf P A} = {\bf L U}$$. It consists of the following steps.

1. First choose $$i$$ so that row $$i$$ in $${\bf A}$$ has the largest absolute first entry. That is, $$\vert A_{i1}\vert \ge \vert A_{j1}\vert$$ for all $$j$$. Let $${\bf P}_1$$ be the permutation matrix that pivots (shifts) row $$i$$ to the first row, and leaves all other rows in order. We can explicitly write $${\bf P}_1$$ as

    $$
    {\bf P}_1 =
    \begin{bmatrix}
    0_{1(i-1)}     & 1 & 0_{1(n-i)} \\
    I_{(i-1)(i-1)} & 0 & 0_{(i-1)(n-i)} \\
    0_{(n-i)(i-1)} & 0 & I_{(n-i)(n-i)}
    \end{bmatrix}
    =
    \begin{bmatrix}
    0      & \ldots & 0      & 1      & 0      & \ldots & 0 \\
    1      & \ldots & 0      & 0      & 0      & \ldots & 0 \\
    \vdots & \ddots & \vdots & \vdots & \vdots & \ldots & 0 \\
    0      & \ldots & 1      & 0      & 0      & \ldots & 0 \\
    0      & \ldots & 0      & 0      & 1      & \ldots & 0 \\
    \vdots & \ddots & \vdots & \vdots & \vdots & \ldots & 0 \\
    0      & \ldots & 0      & 0      & 0      & \ldots & 1
    \end{bmatrix}.
    $$

2. Write $$\bar{ {\bf A} }$$ to denote the pivoted $${\bf A}$$ matrix, so $$\bar{ {\bf A} } = {\bf P}_1 {\bf A}$$.

3. Let $${\bf P}_2$$ be a permutation matrix that leaves the first row where it is, but permutes all other rows. We can write $${\bf P}_2$$ as
    $$
    {\bf P}_2 =
    \begin{bmatrix}
    1              & \boldsymbol{0} \\
    \boldsymbol{0} & P_{22}
    \end{bmatrix},
    $$
where $${\bf P}_{22}$$ is an $$(n-1) \times (n-1)$$ permutation matrix.

4. Factorize the (unknown) full permutation matrix $${\bf P}$$ as the product of $${\bf P}_2$$ and $${\bf P}_1$$, so $${\bf P} = {\bf P}_2 {\bf P}_1$$. This means that $${\bf P} A = {\bf P}_2 {\bf P}_1 A = {\bf P}_2 \bar{ {\bf A} }$$, which first shifts row $$i$$ of $${\bf A}$$ to the top, and then permutes the remaining rows. This is a completely general permutation matrix $${\bf P}$$, but this factorization is key to enabling a recursive algorithm.

5. Using the factorization $${\bf P} = {\bf P}_2 {\bf P}_1$$, now write the LUP factorization in block form as

    $$
    \begin{aligned}
    {\bf P A} &= {\bf L U} \\
    {\bf P_2} \bar{ {\bf A} } &= {\bf L U} \\
    \begin{bmatrix}
    1              & \boldsymbol{0} \\
    \boldsymbol{0} & {\bf P}_{22}
    \end{bmatrix}
    \begin{bmatrix}
    \bar{a}_{11} & \bar{\boldsymbol{a}}_{12} \\
    \bar{\boldsymbol{a}}_{21} & \bar{ {\bf A} }_{22}
    \end{bmatrix}
    &=
    \begin{bmatrix}
    1 & \boldsymbol{0} \\
    \boldsymbol{\ell}_{21} & {\bf L}_{22}
    \end{bmatrix}
    \begin{bmatrix}
    u_{11} & \boldsymbol{u}_{12} \\
    \boldsymbol{0} & {\bf U}_{22}
    \end{bmatrix}
    \\
    \begin{bmatrix}
    \bar{a}_{11} & \bar{\boldsymbol{a}}_{12} \\
    {\bf P}_{22} \bar{\boldsymbol{a}}_{21} & {\bf P}_{22} \bar{ {\bf A} }_{22}
    \end{bmatrix}
    &=
    \begin{bmatrix}
    u_{11} & \boldsymbol{u}_{12} \\
    u_{11} \boldsymbol{\ell}_{21} & (\boldsymbol{\ell}_{21} \boldsymbol{u}_{12} + {\bf L}_{22} {\bf U}_{22})
    \end{bmatrix}
    \end{aligned}
    $$

6. Equating the entries in the above matrices gives the equations

    $$
    \begin{aligned}
    \bar{a}_{11} &= u_{11} \\
    \bar{\boldsymbol{a}}_{12} &= \boldsymbol{u}_{12} \\
    {\bf P}_{22} \bar{\boldsymbol{a}}_{21} &= u_{11} \boldsymbol{\ell}_{21} \\
    {\bf P}_{22} \bar{A}_{22} &= \boldsymbol{\ell}_{21} \boldsymbol{u}_{12} + {\bf L}_{22} {\bf U}_{22}.
    \end{aligned}
    $$

7. Substituting the first three equations above into the last one and rearranging gives

    $$
    {\bf P}_{22} \underbrace{\Bigl(\bar{A}_{22} - \bar{\boldsymbol{a}}_{21} (\bar{a}_{11})^{-1} \bar{\boldsymbol{a}}_{12}\Bigr)}_{\text{Schur complement } {\bf S}_{22}} = {\bf L}_{22} {\bf U}_{22}.
    $$

8. Recurse to find the LUP decomposition of $$S_{22}$$, resulting in $${\bf L}_{22}$$, $${\bf U}_{22}$$, and $${\bf P}_{22}$$ that satisfy the above equation.

9. Solve for the first rows and columns of $${\bf L}$$ and $${\bf U}$$ with the above equations to give

    $$
    \begin{aligned}
    u_{11} &= \bar{a}_{11} \\
    \boldsymbol{u}_{12} &= \bar{\boldsymbol{a}}_{12} \\
    \boldsymbol{\ell}_{21} &= \frac{1}{\bar{a}_{11}} {\bf P}_{22} \bar{\boldsymbol{a}}_{21}.
    \end{aligned}
    $$

10. Finally, reconstruct the full matrices $${\bf L}$$, $${\bf U}$$, and $${\bf P}$$ from the component parts.

In code the **_recursive leading-row-column LUP algorithm_** for finding the LU decomposition of $${\bf A}$$ with partial pivoting is:

```python
import numpy as np
def lup_decomp(A):
    """(L, U, P) = lup_decomp(A) is the LUP decomposition P A = L U
       A is any matrix
       L will be a lower-triangular matrix with 1 on the diagonal, the same shape as A
       U will be an upper-triangular matrix, the same shape as A
       P will be a permutation matrix, the same shape as A
    """
    n = A.shape[0]
    if n == 1:
        L = np.array([[1]])
        U = A.copy()
        P = np.array([[1]])
        return (L, U, P)

    i = np.argmax(A[:,0])
    A_bar = np.vstack([A[i,:], A[:i,:], A[(i+1):,:]])

    A_bar11 = A_bar[0,0]
    A_bar12 = A_bar[0,1:]
    A_bar21 = A_bar[1:,0]
    A_bar22 = A_bar[1:,1:]

    S22 = A_bar22 - np.dot(A_bar21, A_bar12) / A_bar11

    (L22, U22, P22) = lup_decomp(S22)

    L11 = 1
    U11 = A_bar11

    L12 = np.zeros(n-1)
    U12 = A_bar12.copy()

    L21 = np.dot(P22, A_bar21) / A_bar11
    U21 = np.zeros(n-1)

    L = np.block([[L11, L12], [L21, L22]])
    U = np.block([[U11, U12], [U21, U22]])
    P = np.block([
        [np.zeros((1, i-1)), 1,                  np.zeros((1, n-i))],
        [P22[:,:(i-1)],      np.zeros((n-1, 1)), P22[:,i:]]
    ])
    return (L, U, P)
```

The properties of the recursive leading-row-column LUP decomposition algorithm are:

1. The computational complexity (number of operations) of the algorithm is $$\mathcal{O}(n^3)$$ as $$n \to \infty$$.

2. The last step in the code that computes $${\bf P}$$ does not do so by constructing and multiplying $${\bf P}_2$$ and $${\bf P}_1$$. This is because this would be an $$\mathcal{O}(n^3)$$ step, making the whole algorithm $$\mathcal{O}(n^4)$$. Instead we take advantage of the special structure of $${\bf P}_2$$ and $${\bf P}_1$$ to compute $${\bf P}$$ with $$\mathcal{O}(n^2)$$ work.

## Solving General Linear Systems using LUP Decomposition

Just as with the plain LU decomposition, we can use LUP decomposition to solve the linear system $${\bf A x }= {\bf b}$$. This is the **_linear solver using LUP decomposition_** algorithm.

The properties of this algorithm are:

1. The algorithm may fail.  In particular if $${\bf A}$$ is singular (or singular
   in finite precision), U will have a zero on it's diagonal.
2. As LU factorization has $$\mathcal{O}(n^3)$$ opeations, and solving $$Ly = b$$ and $$Ux = y$$ both have $$\mathcal{O}(n^2)$$ opeations, the number of total operations in this algorithm is $$\bf{\mathcal{O}(n^3)}$$ as $$n \to \infty$$.
3. We decouple the factorization step from the actual solve, since in enigeering applications, it might be possible to reuse the LU factorization of the same matrix in solving multiple linear systems.

The code for the **_linear solver using LUP decomposition_** is:

```python
import numpy as np
def linear_solve(A, b):
    """x = linear_solve(A, b) is the solution to A x = b (computed with partial pivoting)
       A is any matrix
       b is a vector of the same leading dimension as A
       x will be a vector of the same leading dimension as A
    """
    (L, U, P) = lup_decomp(A)
    x = lup_solve(L, U, P, b)
    return x
```

### Example: LUP Decomposition

Consider the matrix
$$
\bf{A} =
\begin{bmatrix}
2 & 1 & 1 & 0 \\
4 & 3 & 3 & 1 \\
8 & 7 & 9 & 5 \\
6 & 7 & 9 & 8
\end{bmatrix}
.$$

How can we find an LUP Decomposition for this matrix?

<details>
    <summary><strong>Answer</strong></summary>

Note that we use \({\bf {\overline{M}}}\) to keep track of the matrix (e.g. \(\bf L_{22} U_{22}\) in the first step) that needs recursive factorization, and the tensor product operator "\(\otimes\)" to denote the outer product of two vectors.

In the first step:
$$
\bf{P\overline{M}} =
\begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
2 & 1 & 1 & 0 \\
4 & 3 & 3 & 1 \\
8 & 7 & 9 & 5 \\
6 & 7 & 9 & 8
\end{bmatrix} = 
\begin{bmatrix}
8 & 7 & 9 & 5 \\
4 & 3 & 3 & 1 \\
2 & 1 & 0 & 0 \\
6 & 7 & 9 & 8
\end{bmatrix}
$$

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
\textbf{0.5} & 1 & 0 & 0 \\
\textbf{0.25} & ? & 1 & 0 \\
\textbf{0.75} & ? & ? & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
8 & \textbf{7} & \textbf{9} & \textbf{5} \\
0 & ? & ? & ? \\
0 & 0 & ? & ? \\
0 & 0 & 0 & ?
\end{bmatrix}; \hspace{5mm}
\bf{\overline{M}} = 
\begin{bmatrix}
3 & 3 & 1 \\
1 & 0 & 0 \\
7 & 9 & 8 
\end{bmatrix} - 
\begin{bmatrix}
0.5 \\
0.25 \\
0.75
\end{bmatrix} \otimes
\begin{bmatrix}
7 \\
9 \\
5
\end{bmatrix} =
\begin{bmatrix}
-0.5 & -1.5 & -1.5 \\
-0.75 & -1.25 & -1.25 \\
1.75 & 2.25 & 4.25 
\end{bmatrix}
$$


Similarly, in the second (recursive) step:

$$
\bf{P\overline{M}} =
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0 
\end{bmatrix}
\begin{bmatrix}
-0.5 & -1.5 & -1.5 \\
-0.75 & -1.25 & -1.25 \\
1.75 & 2.25 & 4.25 
\end{bmatrix} = 
\begin{bmatrix}
1.75 & 2.25 & 4.25 \\
-0.75 & -1.25 & -1.25 \\
-0.5 & -1.5 & -1.5
\end{bmatrix}
$$

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
\textbf{0.75} & 1 & 0 & 0 \\
0.25 & \textbf{-0.428} & 1 & 0 \\
\textbf{0.75} & \textbf{-0.285} & ? & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
8 & 7 & 9 & 5 \\
0 & \textbf{1.75} & \textbf{2.25} & \textbf{4.25} \\
0 & 0 & ? & ? \\
0 & 0 & 0 & ?
\end{bmatrix} 
$$

$$
\bf{\overline{M}} = 
\begin{bmatrix}
-1.25 & -1.25 \\
-1.5 & -1.5
\end{bmatrix} - 
\begin{bmatrix}
-0.428 \\
-0.285
\end{bmatrix} \otimes
\begin{bmatrix}
2.25 \\
4.25
\end{bmatrix} =
\begin{bmatrix}
-0.287 & 0.569 \\
-0.8587 & -0.2887
\end{bmatrix}
$$

The next step is:

$$
\bf{P\overline{M}} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
-0.287 & 0.569 \\
-0.8587 & -0.2887
\end{bmatrix} = 
\begin{bmatrix}
-0.8587 & -0.2887 \\
-0.287 & 0.569
\end{bmatrix}
$$

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0.75 & 1 & 0 & 0 \\
\textbf{0.5} & \textbf{-0.285} & 1 & 0 \\
\textbf{0.25} & \textbf{-0.428} & \textbf{0.334} & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
8 & 7 & 9 & 5 \\
0 & 1.75 & 2.25 & 4.25 \\
0 & 0 & \textbf{-0.86} & \textbf{-0.29} \\
0 & 0 & 0 & ?
\end{bmatrix}; \hspace{5mm}
\bf{\overline{M}} = 
\begin{bmatrix}
0.569
\end{bmatrix} - 
\begin{bmatrix}
0.334
\end{bmatrix} \otimes
\begin{bmatrix}
-0.29
\end{bmatrix} =
\begin{bmatrix}
0.67
\end{bmatrix}
$$

So the final result is:

$$
\textbf{L} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0.75 & 1 & 0 & 0 \\
0.5 & -0.285 & 1 & 0 \\
0.25 & -0.428 & 0.334 & 1
\end{bmatrix}; \hspace{5mm}
\textbf{U} =
\begin{bmatrix}
8 & 7 & 9 & 5 \\
0 & 1.75 & 2.25 & 4.25 \\
0 & 0 & -0.86 & -0.29 \\
0 & 0 & 0 & 0.67
\end{bmatrix}; \hspace{5mm}
\textbf{P} =
\begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0
\end{bmatrix}
$$
</details>

### Example: Matrix for which LUP Decomposition Succeeds but LU Decomposition Fails

Consider a matrix which has no LU decomposition:

$$
{\bf A} = \begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}.
$$

How can we find an LUP Decomposition for this matrix?

<details>
    <summary><strong>Answer</strong></summary>

To find the LUP decomposition of \(\bf A\), we first write the permutation matrix \({\bf P}\) that shifts the second row to the top, so that the top-left entry has the largest possible magnitude. This gives

$$
\overbrace{\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}}^{P}
\overbrace{\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}}^{A}
=
\overbrace{\begin{bmatrix}
2 & 1 \\
0 & 1
\end{bmatrix}}^{\bar{A}}
=
\overbrace{\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}}^{L}
\overbrace{\begin{bmatrix}
2 & 1 \\
0 & 1
\end{bmatrix}}^{U}.
$$
</details>

## Review Questions
<ol>
  <li> Given a factorization <span>\({\bf P A} = {\bf LU}\)</span>, how would you solve the system <span>\({\bf A}\mathbf{x} = \mathbf{b}\)</span>?</li>
  <li> Understand the process of solving a triangular system. Solve an example triangular system.</li>
  <li> Recognize and understand Python code implementing forward substitution, back substitution, and LU factorization.</li>
  <li> When does an LU factorization exist?</li>
  <li> When does an LUP factorization exist?</li>
  <li> What special properties do <span>\({\bf P}\), \({\bf L}\)</span> and <span>\({\bf U}\)</span> have?</li>
  <li> Can we find an LUP factorization of a singular matrix?</li>
  <li> What happens if we try to solve a system <span>\({\bf A}\mathbf{x} = \mathbf{b}\)</span> with a singular matrix <span>\({\bf A}\)</span>?</li>
  <li> Compute the LU factorization of a small matrix by hand.</li>
  <li> Why do we use pivoting when solving linear systems?</li>
  <li> How do we choose a pivot element?</li>
  <li> What effect does a given permutation matrix have when multiplied by another matrix?</li>
  <li> What is the cost of matrix-matrix multiplication?</li>
  <li> What is the cost of computing an LU or LUP factorization?</li>
  <li> What is the cost of forward or back substitution?</li>
  <li> What is the cost of solving <span>\({\bf A}\mathbf{x} = \mathbf{b}\)</span> for a general matrix?</li>
  <li> What is the cost of solving <span>\({\bf A}\mathbf{x} = \mathbf{b}\)</span> for a triangular matrix?</li>
  <li> What is the cost of solving <span>\({\bf A}\mathbf{x} = \mathbf{b_i}\)</span> with the same matrix <span>\({\bf A}\)</span> and several right-hand side vectors <span>\(\mathbf{b}_i\)</span>?</li>
  <li>Given a process that takes time <span>\(\mathcal{O}(n^k)\)</span>, what happens to the runtime if we double the input size (i.e. double <span>\(n\)</span>)? What if we triple the input size?</li>
</ol>
