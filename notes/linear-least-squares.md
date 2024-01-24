---
title: Least Squares Data Fitting
description: Add description here...
sort: 17
---
# Least Squares Data Fitting

* * *

## Learning Objectives

*   Set up a linear least-squares problem from a set of data
*   Use an SVD to solve the least-squares problem
*   Quantify the error in a least-squares problem


## Linear Regression with a Set of Data

Consider a set of <span>\\(m\\)</span> data points (where <span>\\(m>2\\)</span>), \\(\{(t_1,y_1),(t_2,y_2),\dots,(t_m,y_m)\}\\). Suppose we want to find a straight line that best fits these data points.
Mathematically, we are finding $$x_0$$ and $$x_1$$ such that
<div>\[ y_i = x_1\,t_i + x_0, \quad \forall i \in [1,m]. \]</div>

In matrix form, the resulting linear system is
<div>\[ \begin{bmatrix} 1 & t_1 \\ 1& t_2 \\ \vdots & \vdots\\ 1& t_m  \end{bmatrix} \begin{bmatrix} x_0\\ x_1 \end{bmatrix} = \begin{bmatrix} y_1\\ y_2\\ \vdots\\ y_m \end{bmatrix} \]</div>

However, it is obvious that we have more equations than unknowns, and there is usually no exact solution to the above problem.

Generally, if we have a linear system

$${\bf A x} = {\bf b} $$

where $${\bf A}$$ is an \\(m\times n\\) matrix. When <span>\\(m>n\\)</span>  we call this system **_overdetermined_** and the equality is usually not exactly satisfiable as $${\bf b}$$ may not lie in the column space of <span>\\({\bf A}\\)</span>.

Therefore, an overdetermined system is better written as

$${\bf A x} \cong {\bf b} $$

## Linear Least-squares Problem

For an overdetermined system \\({\bf A x}\cong {\bf b}\\), we are typically looking for a solution \\({\bf x}\\) that minimizes the squared Euclidean norm of the residual vector \\({\bf r} = {\bf b} - {\bf A} {\bf x}\\),

$$\min_{ {\bf x} } \|{\bf r}\|_2^2 = \min_{ {\bf x} } \|{\bf b} - {\bf A}  {\bf x}\|_2^2.$$

This problem \\(A {\bf x} \cong {\bf b}\\) is called a **_linear least-squares problem_**, and the solution \\({\bf x}\\) is called **_least-squares solution_**. Linear Least Squares problem \\(A {\bf x} \cong {\bf b}\\) always has solution. Here we will first focus on linear least-squares problems.

## Data Fitting vs Interpolation

It is important to understand that interpolation and least-squares data fitting, while somewhat similar, are fundamentally different in their goals. In both problems we have a set of data points <span>\\((t_i, y_i)\\)</span>, \\(i=1,\ldots,m\\), and we are attempting to determine the coefficients for a linear combination of basis functions.

With interpolation, we are looking for the linear combination of basis functions such that the resulting function passes through each of the data points _exactly_. So, for <span>\\(m\\)</span> unique data points, we need <span>\\(m\\)</span> linearly independent basis functions (and the resulting linear system will be square and full rank, so it will have an exact solution).

In contrast, however, with least squares data fitting we have some model that we are trying to find the parameters of the model that best fits the data points. For example, with linear least squares we may have 300 noisy data points that we want to model as a quadratic function. Therefore, we are trying represent our data as

$$y = x_0 + x_1 t + x_2 t^2 $$

where <span>\\(x_0, x_1,\\)</span> and <span>\\(x_2\\)</span> are the unknowns we want to determine (the coefficients to our basis functions). Because there are significantly more data points than parameters, we do not expect that the function will exactly pass through the data points. For this example, with noisy data points we would not want our function to pass through the data points exactly as we are looking to model the general trend and not capture the noise.

## Normal Equations

Consider the least squares problem, \\({\bf A}  {\bf x} \cong {\bf b}\\), where <span>\\({\bf A} \\)</span> is \\(m \times n\\) real matrix (with <span>\\(m > n\\)</span>). As stated above, the least squares solution minimizes the squared 2-norm of the residual.
Hence, we want to find the \\(\mathbf{x}\\) that minimizes the function:

$$\phi(\mathbf{x}) = \|\mathbf{r}\|_2^2 = (\mathbf{b} - {\bf A} \mathbf{x})^T (\mathbf{b} - {\bf A} \mathbf{x}) = \mathbf{b}^T \mathbf{b} - 2\mathbf{x}^T {\bf A} ^T \mathbf{b} + \mathbf{x}^T {\bf A} ^T {\bf A}  \mathbf{x}.$$

To solve this unconstrained minimization problem, we need to satisfy the first order necessary condition to get a stationary point:

$$ \nabla \phi(\mathbf{x}) = 0 \Rightarrow -2 {\bf A} ^T \mathbf{b} + 2 {\bf A} ^T {\bf A}  \mathbf{x} = 0.$$

The resulting square (\\(n \times n\\)) linear system

$${\bf A} ^T {\bf A}  \mathbf{x} = {\bf A} ^T \mathbf{b}$$

is called the system of **normal equations**. If the matrix $${\bf A} $$ is full rank, the least-squares solution is unique and given by:

$${\bf x} = ({\bf A} ^T {\bf A})^{-1} {\bf A} ^T \mathbf{b}$$

We can look at the second-order sufficient condition of the the minimization problem by evaluating the Hessian of $$\phi$$:

$${\bf H} = 2 {\bf A} ^T {\bf A}$$

Since the Hessian is symmetric and positive-definite, we confirm that the least-squares solution $${\bf x}$$ is indeed a minimizer.

Although the least squares problem can be solved via the normal equations for full rank matrices,
the solution tend to worsen the conditioning of the problem. Specifically,
<div>\[\text{cond}({\bf A}^T {\bf A}) = (\text{cond}({\bf A}))^2.\]</div>

Because of this, finding the least squares solution using Normal Equations is often not a good choice (however, simple to implement).

Another approach to solve Linear Least Squares is to find $${\bf y} = {\bf A} {\bf x}$$ which is closest to the vector $${\bf b}$$.
When the residual $${\bf r} = {\bf b} - {\bf y} = {\bf b} - {\bf A} {\bf x}$$ is orthogonal to all columns of $${\bf A}$$, then $${\bf y}$$ is closest to $${\bf b}$$.

## Computational Complexity

Since the system of normal equations yield a square and symmetric matrix, the least-squares solution can be
computed using efficient methods such as Cholesky factorization. Note that the overall computational complexity of the factorization is
$$\mathcal{O}(n^3)$$. However, the construction of the matrix $${\bf A} ^T {\bf A}$$ has complexity $$\mathcal{O}(mn^2)$$.
In typical data fitting problems, $$ m >> n$$ and hence the overall complexity of the Normal Equations method is $$\mathcal{O}(mn^2)$$.

## Solving Least-Squares Problems Using SVD

Another way to solve the least-squares problem \\({\bf A} {\bf x} \cong {\bf b}\\)
(where we are looking for \\({\bf x}\\) that minimizes $$\|{\bf b} - {\bf A} {\bf x}\|_2^2$$ is to use the singular value decomposition
(SVD) of <span>\\({\bf A}\\)</span>,

$${\bf A} = {\bf U \Sigma V}^T $$

where the squared norm of the residual becomes:

<div>\[ \begin{align} \|{\bf b} - {\bf A} {\bf x}\|_2^2 &= \|{\bf b} - {\bf U \Sigma V}^T {\bf x}\|_2^2 & (1)\\ &= \|{\bf U}^T ({\bf b} - {\bf U \Sigma V}^T {\bf x})\|_2^2 & (2)\\ &= \|{\bf U}^T {\bf b} - {\bf \Sigma V}^T {\bf x}\|_2^2 \end{align} \]</div>

We can go from (1) to (2) because multiplying a vector by an orthogonal matrix does not change the 2-norm of the vector. Now let

<div>\[ {\bf y} = {\bf V}^T {\bf x} \]</div>
<div>\[ {\bf z} = {\bf U}^T {\bf b}, \]</div>
then we are looking for \\({\bf y}\\) that minimizes
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





#### Example of a Least-squares solution using SVD

Assume we have <span>\\(3\\)</span> data points, \\(\{(t_i,y_i)\}=\{(1,1.2),(2,1.9),(3,1)\}\\), we want to find a line that best fits these data points. The code for using SVD to solve this least-squares problem is:

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
while k < len(A[0]) and s[k] > threshold:
  y[k] = z[k]/s[k]
  k += 1

x = np.dot(V,y)
print("The function of the best line is: y = " + str(x[0]) + "x + " + str(x[1]))
```

## Non-linear Least-squares Problem vs. Linear Least-squares Problem

The above linear least-squares problem is associated with an overdetermined linear system \\(A {\bf x} \cong {\bf b}.\\) This problem is called "linear"  because the fitting function we are looking for is linear in the components of \\({\bf x}\\). For example, if we are looking for a polynomial fitting function
<div>\[ f(t,{\bf x}) = x_1 + x_2t + x_3t^2 + \dotsb + x_nt^{n-1} \]</div>
to fit the data points \\(\{(t_i,y_i), i = 1, ...,  m\}\\) and (<span>\\(m > n\\)</span>), the problem can be solved using the linear least-squares method, because \\(f(t,{\bf x})\\) is linear in the components of \\({\bf x}\\) (though \\(f(t,{\bf x})\\) is nonlinear in <span>\\(t\\)</span>).

If the fitting function \\(f(t,{\bf x})\\) for data points $$(t_i,y_i), i = 1, ...,  m$$ is nonlinear in the components of \\({\bf x}\\), then the problem is a non-linear least-squares problem. For example, fitting sum of exponentials
<div>\[ f(t,{\bf x}) = x_1 e^{x_2 t} + x_2 e^{x_3 t} + \dotsb + x_{n-1} e^{x_n t} \]</div>
is a **_non-linear least-squares problem_**.

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-17-least-squares.html)

## ChangeLog

* 2023-04-28 Yuxuan Chen <yuxuan19@illinois.edu>: adding computational complexity using reduced SVD
* 2022-04-09 Arnav Shah <arnavss2@illinois.edu>: add few comments from slides asked in homework
* 2020-08-08 Jerry Yang <jiayiy7@illinois.edu>: adds formal proof link for solving least-squares using SVD
* 2020-04-26 Mariana Silva <mfsilva@illinois.edu>: improved text overall; removed theory of the nonlinear least-squares
* 2018-11-14 Erin Carrier <ecarrie2@illinois.edu>: fix typo in lstsq res sum range
* 2018-01-14 Erin Carrier <ecarrie2@illinois.edu>: removes demo links
* 2017-11-29 Erin Carrier <ecarrie2@illinois.edu>: fixes typos in lst-sq code,
  jacobian desc in nonlinear lst-sq
* 2017-11-17 Erin Carrier <ecarrie2@illinois.edu>: fixes incorrect link
* 2017-11-16 Erin Carrier <ecarrie2@illinois.edu>: adds review questions
  minor formatting changes throughout for consistency,
  adds normal equations and interp vs lst-sq sections
  removes Gauss-Newton from nonlinear least squares
* 2017-11-12 Yu Meng <yumeng5@illinois.edu>: first complete draft
* 2017-10-17 Luke Olson <lukeo@illinois.edu>: outline
