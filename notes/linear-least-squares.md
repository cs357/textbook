---
title: Least Squares Fitting
description: Solving Least Squares problems with different methods
sort: 17
author:
  - CS 357 Course Staff
changelog:
  - 
    name: Bhargav Chandaka
    netid: bhargav9
    date: 2024-03-06
    message: major reorganziation to match up with content in slides/videos
  - 
    name: Yuxuan Chen
    netid: yuxuan19
    date: 2023-04-28
    message: adding computational complexity using reduced SVD
  - 
    name: Arnav Shah
    netid: arnavss2
    date: 2022-04-09
    message: add few comments from slides asked in homework
  - 
    name: Jerry Yang
    netid: jiayiy7
    date: 2020-08-08
    message: adds formal proof link for solving least-squares using SVD
  - 
    name: Mariana Silva
    netid: mfsilva
    date: 2020-04-26
    message: improved text overall; removed theory of the nonlinear least-squares
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-11-14
    message: fix typo in lstsq res sum range
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-01-14
    message: removes demo links
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-29
    message: fixes typos in lst-sq code, jacobian desc in nonlinear lst-sq
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-17
    message: fixes incorrect link
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-16
    message: adds review questions, minor formatting changes throughout for consistency, adds normal equations and interp vs lst-sq sections,  removes Gauss-Newton from nonlinear least squares
  - 
    name: Yu Meng
    netid: yumeng5
    date: 2017-11-12
    message: first complete draft
  - 
    name: Luke Olson
    netid: lukeo
    date: 2017-10-17
    message: outline
---

## Learning Objectives

*   Set up a linear least-squares problem from a set of data
*   Use an SVD to solve the least-squares problem
*   Quantify the error in a least-squares problem


## Linear Regression with a Set of Data

Given <span>\\(m\\)</span> data points (where <span>\\(m>2\\)</span>), \\(\\{(t_1,y_1),(t_2,y_2),\dots,(t_m,y_m)\\}\\), we want to find a straight line that best fits these data points.
Mathematically, we are finding $$x_0$$ and $$x_1$$ such that
<div>\[ y_i = x_0 + x_1\,t_i , \quad \forall i \in [1,m]. \]</div>

In matrix form, the resulting linear system is:
<div>\[ \begin{bmatrix} 1 & t_1 \\ 1& t_2 \\ \vdots & \vdots\\ 1& t_m  \end{bmatrix} \begin{bmatrix} x_0\\ x_1 \end{bmatrix} = \begin{bmatrix} y_1\\ y_2\\ \vdots\\ y_m \end{bmatrix} \]</div>

$${\bf A x} = {\bf b} $$

where $${\bf A}$$ is an \\(m\times n\\) matrix, $${\bf x}$$ is an \\(n\times 1\\) matrix, and $${\bf b}$$ is an \\(m\times 1\\) matrix. 

Ideally, we want to find the appropriate linear combination of the columns of $${\bf A}$$ that makes up the vector $${\bf b}$$. If a solution exists that satisfies $${\bf A x} = {\bf b} $$, then $${\bf b} \in range({\bf A})$$.

However, in this system of linear equations, we have more equations than unknowns, and there is usually no exact solution to the above problem.

When \\( m>n\\), we call this linear system **_overdetermined_** and the $${\bf A x} = {\bf b} $$ equality is usually not exactly satisfiable as $${\bf b}$$ may not lie in the column space of $${\bf A}$$.

Therefore, an overdetermined system is better written as

$${\bf A x} \cong {\bf b} $$

## Linear Least-squares Problem

For an overdetermined system \\({\bf A x}\cong {\bf b}\\), we are typically looking for a solution \\({\bf x}\\) that minimizes the Euclidean norm of the residual vector \\({\bf r} = {\bf b} - {\bf A} {\bf x}\\),

$$\min_{ {\bf x} } \|{\bf r}\|_2^2 = \min_{ {\bf x} } \|{\bf b} - {\bf A}  {\bf x}\|_2^2.$$

This problem \\(A {\bf x} \cong {\bf b}\\) is called a **_linear least-squares problem_**, and the solution \\({\bf x}\\) is called the **_least-squares solution_**. $${\bf A}$$ is an $${m \times n}$$ matrix where $${m \ge n}$$,  $${m}$$ is the number of data pair points and $${n}$$ is the number of parameters of the "best fit" function. The Linear Least Squares problem, \\(A {\bf x} \cong {\bf b}\\), **_always_** has a solution. This solution is unique if and only if $${rank({\bf A})= n}$$. 

## Normal Equations

Consider the least squares problem \\({\bf A}  {\bf x} \cong {\bf b}\\), where <span>\\({\bf A} \\)</span> is \\(m \times n\\) real matrix (with <span>\\(m > n\\)</span>). As stated above, the least squares solution minimizes the squared 2-norm of the residual.
Hence, we want to find the \\(\mathbf{x}\\) that minimizes the function:

$$\phi(\mathbf{x}) = \|\mathbf{r}\|_2^2 = (\mathbf{b} - {\bf A} \mathbf{x})^T (\mathbf{b} - {\bf A} \mathbf{x}) = \mathbf{b}^T \mathbf{b} - 2\mathbf{x}^T {\bf A} ^T \mathbf{b} + \mathbf{x}^T {\bf A} ^T {\bf A}  \mathbf{x}.$$

To solve this unconstrained minimization problem, we need to satisfy the first order necessary condition to get a stationary point:

$$ \nabla \phi(\mathbf{x}) = 0 \Rightarrow -2 {\bf A} ^T \mathbf{b} + 2 {\bf A} ^T {\bf A}  \mathbf{x} = 0.$$

The resulting square (\\(n \times n\\)) linear system

$${\bf A} ^T {\bf A}  \mathbf{x} = {\bf A} ^T \mathbf{b}$$

is called the system of **normal equations**. If the matrix $${\bf A} $$ is full rank, the least-squares solution is unique and given by:

$${\bf x} = ({\bf A} ^T {\bf A})^{-1} {\bf A} ^T \mathbf{b}$$

We can look at the second-order sufficient condition of the minimization problem by evaluating the Hessian of $$\phi$$:

$${\bf H} = 2 {\bf A} ^T {\bf A}$$

Since the Hessian is symmetric and positive-definite, we confirm that the least-squares solution $${\bf x}$$ is indeed a minimizer.

Although the least squares problem can be solved via the normal equations for full rank matrices,
the solution tend to worsen the conditioning of the problem. Specifically,
<div>\[\text{cond}({\bf A}^T {\bf A}) = (\text{cond}({\bf A}))^2.\]</div>

Because of this, finding the least squares solution using Normal Equations is often not a good choice (however, simple to implement).

Another approach to solve Linear Least Squares is to find $${\bf y} = {\bf A} {\bf x}$$ which is closest to the vector $${\bf b}$$.
When the residual $${\bf r} = {\bf b} - {\bf y} = {\bf b} - {\bf A} {\bf x}$$ is orthogonal to all columns of $${\bf A}$$, then $${\bf y}$$ is closest to $${\bf b}$$.

$${\bf A}^T{\bf r} = {\bf A^T}\left({\bf b} - {\bf A} {\bf x}\right) = 0 \implies {\bf A} ^T {\bf A}  {\bf x} = {\bf A}^T  {\bf b}$$

## Data Fitting vs Interpolation

It is important to understand that interpolation and least-squares data fitting, while somewhat similar, are fundamentally different in their goals. In both problems we have a set of data points <span>\\((t_i, y_i)\\)</span>, \\(i=1,\ldots,m\\), and we are attempting to determine the coefficients for a linear combination of basis functions.

With interpolation, we are looking for the linear combination of basis functions such that the resulting function passes through each of the data points _exactly_. So, for <span>\\(m\\)</span> unique data points, we need <span>\\(m\\)</span> linearly independent basis functions (and the resulting linear system will be square and full rank, so it will have an exact solution).

In contrast, however, with least squares data fitting we have some model that we are trying to find the parameters of the model that best fits the data points. For example, with linear least squares we may have 300 noisy data points that we want to model as a quadratic function. Therefore, we are trying represent our data as

$$y = x_0 + x_1 t + x_2 t^2 $$

where <span>\\(x_0, x_1,\\)</span> and <span>\\(x_2\\)</span> are the unknowns we want to determine (the coefficients to our basis functions). Because there are significantly more data points than parameters, we do not expect that the function will exactly pass through the data points. For this example, with noisy data points we would not want our function to pass through the data points exactly as we are looking to model the general trend and not capture the noise.

## Computational Complexity

Since the system of normal equations yield a square and symmetric matrix, the least-squares solution can be computed using efficient methods such as Cholesky factorization. Note that the overall computational complexity of the factorization is $$\mathcal{O}(n^3)$$. However, the construction of the matrix $${\bf A} ^T {\bf A}$$ has complexity $$\mathcal{O}(mn^2)$$.

In typical data fitting problems, $$ m >> n$$, so the overall time complexity of the Normal Equations method is $${\bf \mathcal{O}(mn^2)}$$.

## Solving Least-Squares Problems Using SVD

Another way to solve the least-squares problem \\({\bf A} {\bf x} \cong {\bf b}\\)
(where we are looking for \\({\bf x}\\) that minimizes $$\|{\bf b} - {\bf A} {\bf x}\|_2^2$$) is to use the singular value decomposition
(SVD) of <span>\\({\bf A}\\)</span>,

$${\bf A} = {\bf U \Sigma V}^T $$

where the squared norm of the residual becomes:

<div>\[ \begin{align} \|{\bf b} - {\bf A} {\bf x}\|_2^2 &= \|{\bf b} - {\bf U \Sigma V}^T {\bf x}\|_2^2 & (1)\\ &= \|{\bf U}^T ({\bf b} - {\bf U \Sigma V}^T {\bf x})\|_2^2 & (2)\\ &= \|{\bf U}^T {\bf b} - {\bf \Sigma V}^T {\bf x}\|_2^2 \end{align} \]</div>

We can go from (1) to (2) because multiplying a vector by an orthogonal matrix does not change the 2-norm of the vector. Now, let

<div>\[ {\bf y} = {\bf V}^T {\bf x} \]</div>
<div>\[ {\bf z} = {\bf U}^T {\bf b}, \]</div>
We are now looking for \\({\bf y}\\) that minimizes
<div>\[ \|{\bf z} - \Sigma {\bf y}\|_2^2. \]</div>

Note that
<div>\[ \Sigma {\bf y} = \begin{bmatrix} \sigma_1 y_1\\ \sigma_2 y_2\\ \vdots \\ \sigma_n y_n \end{bmatrix},(\sigma_i = \Sigma_{i,i}) \]</div>
so we choose
<div>\[ y_i = \begin{cases} \frac{z_i}{\sigma_i} & \sigma_i \neq 0\\ 0 & \sigma_i = 0 \end{cases} \]</div>
which will minimize $$\|{\bf z} - {\bf \Sigma} {\bf y}\|_2^2$$. Finally, we compute

$$ {\bf x} = {\bf V} {\bf y} $$

to find \\({\bf x}\\). The expression of least-squares solution is

$$ {\bf x} = \sum_{\sigma_i \neq 0} \frac{ {\bf u}_i^T {\bf b} }{\sigma_i} {\bf v}_i $$

where \\({\bf u}_i\\) represents the <span>\\(i\\)</span>th column of <span>\\({\bf U}\\)</span> and \\({\bf v}_i\\) represents the <span>\\(i\\)</span>th column of <span>\\({\bf V}\\)</span>.

In closed-form, we can express the least-squares solution as:

$$ {\bf x} = {\bf V\Sigma}^{+}{\bf U}^T{\bf b}$$

where \\({\bf \Sigma}^{+}\\) is the pseudoinverse of the singular matrix computed by taking the reciprocal of the non-zero diagonal entries, leaving the zeros in place and transposing the resulting matrix. For example:

$$\Sigma = \begin{bmatrix} \sigma_1 & & &\\ & \ddots & &  \\&  & \sigma_r &\\
&  &  & 0\\ 0 & ... & ... & 0 \\ \vdots & \ddots & \ddots & \vdots \\ 0 & ... & ... & 0 \end{bmatrix}
\quad \implies \quad
\Sigma^{+} = \begin{bmatrix} \frac{1}{\sigma_1} & & & & 0 & \dots & 0 \\ & \ddots & & & & \ddots &\\ & & \frac{1}{\sigma_r} & & 0 & \dots & 0 \\ & & & 0 & 0 & \dots & 0 \end{bmatrix}.$$

Or in reduced form:

$$ {\bf x} = {\bf V\Sigma}_R^{+}{\bf U}_R^T{\bf b}$$

## Computational Complexity Using Reduced SVD

Solving the least squares problem using a **given** reduced SVD has time complexity  <span>\\(\mathcal{O}(mn)\\)</span>.

Assume we know the reduced SVD of $$\bf A$$ beforehand (which inplies that we know the entries of $$\bf V, \Sigma^{+}_{R}, U^{T}_{R}$$ beforehand). 

Then, solving $$\bf z = U^{T}_{R}b$$ with $$n \times m $$ matrix $$\bf U^{T}_{R}$$ and $$m \times 1 $$ vector $$ \bf b$$ has $$\mathcal{O}(mn)$$.

Solving $$\bf y = \Sigma^{+}_{R}z$$ with $$n \times n $$ matrix $$\bf \Sigma^{+}_{R}$$ and $$n \times 1 $$ vector $$ \bf z$$ has $$\mathcal{O}(n)$$ since $$\bf \Sigma^{+}_{R}$$ is a diagonal matrix.

Solving $$\bf x = Vy$$ with $$n \times n $$ matrix $$\bf V$$ and $$n \times 1 $$ vector $$ \bf y$$ has $$\mathcal{O}(n^2)$$.

Therefore the time complexity is $$\mathcal{O}(n^2 + n + mn) = \mathcal{O}(mn)$$ since $$m > n$$. We can achieve this only when we know the Reduced SVD of $$\bf A$$ beforehand.

## Determining Residual in Least-Squares Problem Using SVD

We've shown above how the SVD can be used to find the least-squares solution (the solution that minimizes the squared 2-norm of the residual) to the least squares problem \\({\bf A} {\bf x} \cong {\bf b}\\). We can also use the SVD to determine an exact expression for the value of the residual with the least-squares solution.

Assume in the SVD of <span>\\({\bf A}\\)</span>, \\({\bf A} = {\bf U \Sigma V}^T\\), the diagonal entries of \\({\bf \Sigma}\\) are in descending order (\\(\sigma_1 \ge \sigma_2 \ge \dots \\)), and the first <span>\\(r\\)</span> diagonal entries of \\({\bf \Sigma}\\) are nonzeros while all others are zeros, then

$$ \begin{align} \|{\bf b} - A {\bf x}\|_2^2 &= \|{\bf z} - \Sigma {\bf y}\|_2^2\\ &= \sum_{i=1}^n (z_i - \sigma_i y_i)^2\\ &= \sum_{i=1}^r (z_i - \sigma_i \frac{z_i}{\sigma_i})^2 + \sum_{i=r+1}^n (z_i - 0)^2\\ &= \sum_{i=r+1}^n z_i^2 \end{align} $$

Recall that
<div>\[ {\bf z} = {\bf U}^T {\bf b}, \]</div>
we get
<div>\[ \|{\bf b} - {\bf A} {\bf x}\|_2^2 = \sum_{i=r+1}^n ({\bf u}_i^T {\bf b})^2 \]</div>
where \\({\bf u}_i\\) represents the <span>\\(i\\)</span>th column of <span>\\({\bf U}\\)</span>.

(For more formal proof check 
[this video.](https://mediaspace.illinois.edu/media/t/1_w6u83js8/158464041))


### Example of a Least-Squares solution using SVD

Assume we have <span>\\(3\\)</span> data points, \\(\{(t_i,y_i)\}=\{(1,1.2),(2,1.9),(3,1)\}\\), we want to find the coefficients for a line, $${y = x_0 + x_1 t}$$, that best fits these data points. The code for using SVD to solve this least-squares problem is:

```python
import numpy as np
import numpy.linalg as la

A = np.array([[1,1],[2,1],[3,1]])
b = np.array([1.2,1.9,1])
U, s, V = la.svd(A)
V = V.T
y = np.zeros(len(A[0]))
z = np.dot(U.T,b)
k = 0
threshold = 0.01
# matrix multiplying A by pseudo-inverse of sigma
while k < len(A[0]) and s[k] > threshold:
  y[k] = z[k]/s[k]
  k += 1

x = np.dot(V,y)
print("The function of the best line is: y = " + str(x[0]) + "x + " + str(x[1]))
```

## Non-linear Least-Squares Problems vs. Linear Least-Squares Problems

The above linear least-squares problem is associated with an overdetermined linear system \\(A {\bf x} \cong {\bf b}.\\) This problem is called "linear" because the fitting function we are looking for is linear in the components of \\({\bf x}\\). For example, if we are looking for a polynomial fitting function
<div>\[ f(t,{\bf x}) = x_1 + x_2t + x_3t^2 + \dotsb + x_nt^{n-1} \]</div>
to fit the data points \\(\{(t_i,y_i), i = 1, ...,  m\}\\) and (<span>\\(m > n\\)</span>), the problem can be solved using the linear least-squares method, because \\(f(t,{\bf x})\\) is linear in the components of \\({\bf x}\\) (though \\(f(t,{\bf x})\\) is nonlinear in <span>\\(t\\)</span>).

If the fitting function \\(f(t,{\bf x})\\) for data points $$(t_i,y_i), i = 1, ...,  m$$ is nonlinear in the components of \\({\bf x}\\), then the problem is a non-linear least-squares problem. For example, fitting sum of exponentials
<div>\[ f(t,{\bf x}) = x_1 e^{x_2 t} + x_2 e^{x_3 t} + \dotsb + x_{n-1} e^{x_n t} \]</div>
is a **_non-linear least-squares problem_**.

Linear least-squares problems have a closed form solution while nonlinear least-squares problems do not have a closed form solution and requires numerical methods to solve.

## Review Questions
1. What value does the least-squares solution minimize?
2. For a given model and given data points, can you form the system $${\bf A x} \cong {\bf b} $$ for a least squares problem?
3. For a small problem, given some data points and a model, can you determine the least squares solution?
4. In general, what can we say about the value of the residual for the least squares solution?
5. What are the differences between least squares data fitting and interpolation?
6. Given the SVD of a matrix $${\bf A}$$, how can we use the SVD to compute the residual of the least squares solution?
7. Given the SVD of a matrix $${\bf A}$$, how can we use the SVD to compute the least squares solution? Be able to do this for a small problem.
8. Given an already computed SVD of a matrix $${\bf A}$$, what is the cost of using the SVD to solve a least squares problem?
9. Why would you use the SVD instead of normal equations to find the solution to $${\bf A x} \cong {\bf b} $$?
10. Which costs less: solving a least squares problem via the normal equations or solving a least squares problem using the SVD?
11. What is the difference between a linear and a nonlinear least squares problem? What sort of model makes it a nonlinear problem? For data points 
$${\left(t_i, y_i\right)}$$, is fitting $${y = a*cos(t) + b}$$ where $${a}$$ and $${b}$$ are the coefficients we are trying to determine a linear or nonlinear least squares problem?