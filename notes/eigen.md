---
title: Eigenvalues and Eigenvectors
description: 
sort: 12
---

# Eigenvalues and Eigenvectors

* * *

## Learning Objectives

*   Compute eigenvalue/eigenvector for various applications.
*   Use the Power Method to find an eigenvector.
*   Definition of eigenvalue/eigenvectors.
*   Methods of obtaining eigenvalues.
*   Computing eigenvalue/eigenvectors for various applications.
*   Using the Power Method to find an eigenvector.


## Eigenvalues and Eigenvectors

An **_eigenvalue_** of an $$n \times n$$ matrix $$\mathbf{A}$$ is a scalar $$\lambda$$ such that
$$ \mathbf{A} {\bf x} = \lambda {\bf x} $$
for some non-zero vector $${\bf x}$$. The eigenvalue $$\lambda$$ can be any real or complex scalar, (which we write $$\lambda \in \mathbb{R}\ \text{or } \lambda \in \mathbb{C}$$). Eigenvalues can be complex even if all the entries of the matrix $$\mathbf{A}$$ are real. In this case, the corresponding vector $${\bf x}$$ must have complex-valued components (which we write $${\bf x}\in \mathbb{C}^n$$). The equation $$\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$$ is called the **_eigenvalue equation_** and any such non-zero vector $${\bf x}$$ is called an **_eigenvector_** of $$\mathbf{A}$$ corresponding to $$\lambda$$.

The eigenvalue equation can be rearranged to $$(\mathbf{A} - \lambda {\bf I}) {\bf x} = 0$$, and because $${\bf x}$$ is not zero this has solutions if and only if $$\lambda$$ is a solution of the **_characteristic equation_**:

$$ \operatorname{det}(\mathbf{A} - \lambda {\bf I}) = 0. $$

The expression $$p(\lambda) = \operatorname{det}(\mathbf{A} - \lambda {\bf I})$$ is called the **_characteristic polynomial_** and is a polynomial of degree <span>$$n$$</span>.

Although all eigenvalues can be found by solving the characteristic equation, there is no general, closed-form analytical solution for the roots of polynomials of degree $$n \ge 5$$ and this is not a good numerical approach for finding eigenvalues.

Unless otherwise specified, we write eigenvalues ordered by magnitude, so that

$$ |\lambda_1| \geq |\lambda_2| \geq \cdots \geq |\lambda_n|, $$

and we normalize eigenvectors, so that $$\|{\bf x}\| = 1$$.

#### Example

First, we find the eigenvalues by solving for the characteristic polynomial.

$$ \bf{A}=\begin{bmatrix} 2 & 1 \\ 4 & 2 \end{bmatrix} \qquad \text{det}(\bf{A}- \bf{I}\lambda)=p(\lambda)=(2-\lambda)^2-4 \rightarrow \lambda_1=4, \lambda_2=0$$

Second, we find the eigenvectors for each eigenvalue by solving for the trivial solution (the nullspace) of $$\bf A-\bf I\lambda$$. **Note** any multiple of $$\bf x$$ below is a valid eigenvector to its eigenvalue.

$$\lambda_1: \begin{bmatrix} 2 & 1 \\ 4 & 2 \end{bmatrix}\bf x=0 \rightarrow \bf{x}=\begin{bmatrix} 1  \\  2 \end{bmatrix} \qquad \lambda_1: \begin{bmatrix} -2 & 1 \\ 4 & -2 \end{bmatrix}\bf{x}=0 \rightarrow \bf{x}=\begin{bmatrix} 1  \\  -2 \end{bmatrix}$$

#### Code Example
The following code snippet finds and prints the eigenvalues and corresponding eigenvectors of a matrix. Take careful note that eigenvectors are stored as columns of a 2d numpy array.

```python
import numpy as np
import numpy.linalg as la
def solve(A):
    # A: nxn matrix
    evals, evecs = la.eig(A)
    for ev in evals:
      print(ev) 
    for i in range(np.shape(evecs)[1]): # print column-wise
      print(evecs[:, i]) 
```

## Diagonalizability

An $$n \times n$$ matrix with <span>$$n$$</span> linearly independent eigenvectors can be expressed as its eigenvalues and eigenvectors as:

$$ \mathbf{A}\mathbf{X} = \begin{bmatrix} \vert & & \vert \\ \lambda_1 {\bf x}_1 & \dots & \lambda_n {\bf x}_n \\ \vert & & \vert \end{bmatrix} = \begin{bmatrix} \vert & & \vert \\ {\bf x}_1 & \dots & {\bf x}_n \\ \vert & & \vert \end{bmatrix} \begin{bmatrix}\lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{bmatrix} = \mathbf{X}\mathbf{D}$$

The eigenvector matrix can be inverted to obtain the following **_similarity transformation_** of $$\mathbf{A}$$:

$$\mathbf{AX} = \mathbf{XD} \iff \mathbf{A} = \mathbf{XDX}^{-1} \iff \mathbf{X}^{-1}\mathbf{A}\mathbf{X} = \mathbf{D} $$

Multiplying the matrix $$\mathbf{A}$$ by $$\mathbf{X}^{-1}$$ on the left and $$\mathbf{X}$$ on the right transforms it into a diagonal matrix; it has been ''diagonalized''.

#### Example: Matrix that is diagonalizable

An $$n \times n$$ matrix is diagonalizable if and only if it has <span>$$n$$</span> linearly independent eigenvectors. For example:

$$ \overbrace{\begin{bmatrix} 1/6 & -1/3 & 1/6 \\ -1/2 & 0 & 1/2 \\ 1/3 & 1/3 & 1/3 \end{bmatrix}}^{\mathbf{X}^{-1}} \overbrace{\begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}}^{\mathbf{A}} \overbrace{\begin{bmatrix} 1 & -1 & 1 \\ -2 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix}}^{\mathbf{X}} = \overbrace{\begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 2 \end{bmatrix}}^{\mathbf{D}} $$

#### Example: Matrix that is not diagonalizable

A matrix $$\mathbf{A}$$ with linearly dependent eigenvectors is not diagonalizable. For example, while it is true that

$$ \overbrace{\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}}^{\mathbf{A}} \overbrace{\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}}^{\mathbf{X}} = \overbrace{\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}}^{\mathbf{X}} \overbrace{\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}}^{\mathbf{D}}, $$

the matrix $$\mathbf{X}$$ does not have an inverse, so we cannot diagonalize $$\mathbf{A}$$ by applying an inverse. In fact, for any non-singular matrix $$\mathbf{P}$$, the product $$\mathbf{P}^{-1}\mathbf{AP}$$ is not diagonal.

#### Things to Remember About Eigenvalues

*   Eigenvalues can have zero value.
*   Eigenvalues can be negative.
*   Eigenvalues can be real or complex numbers.
*   An $$n \times n$$ real matrix can have complex eigenvalues.
*   The eigenvalues of an $$n \times n$$ matrix are not necessarily unique. In fact, we can define the multiplicity of an eigenvalue.
*   If an $$n \times n$$ matrix has $$n$$ linearly independent eigenvectors, then the matrix is diagonalizable.

## Eigenvalues of a Shifted Matrix

Given a matrix $$\mathbf{A}$$, for any constant scalar $$\sigma$$, we define the **_shifted matrix_** is $$\mathbf{A} - \sigma {\bf I}$$. If $$\lambda$$ is an eigenvalue of $$\mathbf{A}$$ with eigenvector $${\bf x}$$ then $$\lambda - \sigma$$ is an eigenvalue of the shifted matrix with the same eigenvector. This can be derived by

$$ \begin{aligned} (\mathbf{A} - \sigma {\bf I}) {\bf x} &= \mathbf{A} {\bf x} - \sigma {\bf I} {\bf x} \\ &= \lambda {\bf x} - \sigma {\bf x} \\ &= (\lambda - \sigma) {\bf x}. \end{aligned} $$

## Eigenvalues of an Inverse

An invertible matrix cannot have an eigenvalue equal to zero. Furthermore, the eigenvalues of the inverse matrix are equal to the inverse of the eigenvalues of the original matrix:

$$ \mathbf{A} {\bf x} = \lambda {\bf x}\implies \\ \mathbf{A}^{-1} \mathbf{A} {\bf x} = \lambda \mathbf{A}^{-1} {\bf x} \implies \\ {\bf x} = \lambda \mathbf{A}^{-1} {\bf x}\implies \\ \mathbf{A}^{-1} {\bf x} = \frac{1}{\lambda} {\bf x}.$$

## Eigenvalues of a Shifted Inverse

Similarly, we can describe the eigenvalues for shifted inverse matrices as:

$$ (\mathbf{A} - \sigma {\bf I})^{-1} {\bf x} = \frac{1}{\lambda - \sigma} {\bf x}.$$

It is important to note here, that the eigenvectors remain unchanged for shifted or/and inverted matrices.

## Expressing an Arbitrary Vector as a Linear Combination of Eigenvectors

If an $$n\times n$$ matrix $$\mathbf{A}$$ is diagonalizable, then we can write an arbitrary vector as a linear combination of the eigenvectors of $$\mathbf{A}$$. Let $${\bf u}_1,{\bf u}_2,\dots,{\bf u}_n$$ be <span>$$n$$</span> linearly independent eigenvectors of $$\mathbf{A}$$; then an arbitrary vector $$\mathbf{x}_0$$ can be written:

$$ {\bf x}_0 = \alpha_1 {\bf u}_1 + \alpha_2 {\bf u}_2 + \dots + \alpha_n {\bf u}_n.$$

If we apply the matrix $$\mathbf{A}$$ to $$\mathbf{x}_0$$:


$$ \begin{align} \mathbf{A}{\bf x}_0 &= \alpha_1 \mathbf{A}{\bf u}_1 + \alpha_2\mathbf{A}{\bf u}_2 + \dots + \alpha_n \mathbf{A}{\bf u}_n, \\ &= \alpha_1 \lambda_1 {\bf u}_1 + \alpha_2\lambda_2 {\bf u}_2 + \dots + \alpha_n \lambda_n {\bf u}_n, \\ &= \lambda_1 \left(\alpha_1 {\bf u}_1 + \alpha_2\frac{\lambda_2}{\lambda_1}{\bf u}_2 + \dots + \alpha_n \frac{\lambda_n}{\lambda_1} {\bf u}_n\right). \\ \end{align}$$

If we repeatedly apply $$\mathbf{A}$$ we have

$$ \mathbf{A}^k{\bf x}_0= \lambda_1^k \left(\alpha_1 {\bf u}_1 + \alpha_2\left(\frac{\lambda_2}{\lambda_1}\right)^k{\bf u}_2 + \dots + \alpha_n \left(\frac{\lambda_n}{\lambda_1}\right)^k {\bf u}_n\right). $$

In the case where one eigenvalue has magnitude that is strictly greater than all the others, i.e.

$$\vert\lambda_1\vert > \vert\lambda_2\vert\geq \vert\lambda_3\vert \geq \dots \geq\vert\lambda_n\vert$$, 

this implies

$$ \lim_{k\to\infty}\frac{\mathbf{A}^k {\bf x}_0}{\lambda_1^{k}} = \alpha_1 {\bf u}_1.$$

This observation motivates the algorithm known as **_power iteration_**, which is the topic of the next section.

## Power Iteration algorithm

For a matrix $${\bf A}$$, power iteration will find a scalar multiple of an eigenvector $${\bf u}_1$$, corresponding to the dominant eigenvalue (largest in magnitude) $$\lambda_1$$, provided that $$\vert\lambda_1\vert$$ is strictly greater than the magnitude of the other eigenvalues, i.e., $$\vert\lambda_1\vert > \vert\lambda_2\vert \ge \dots \ge \vert\lambda_n\vert$$.

Suppose

$$\mathbf{x}_0 = \alpha_1 \mathbf{u}_1 + \alpha_2\mathbf{u}_2 + \dots \alpha_n\mathbf{u}_n,\text{ with }\alpha_1 \neq 0. $$

From the previous section, the iterative sequence

$$\mathbf{x}_k = \mathbf{A}\mathbf{x}_{k-1}\text{ for }k=1,2,3,\dots$$

satisfies

$$ \mathbf{x}_k = \mathbf{A}^k\mathbf{x}_0 \implies \lim_{k\to\infty}\frac{\mathbf{x}_k}{\lambda_1^k} = \alpha_1\mathbf{u}_1. $$

Thus, for large <span>$$k$$</span>, we have $$\mathbf{x}_k\approx \lambda_1^k \alpha_1\mathbf{u}_1$$. Unfortunately, this means that
$$ \|\mathbf{x}_k\| \approx |\lambda_1|^k\cdot\|\alpha_1\mathbf{u}_1\|, $$
which will be very large if $$|\lambda_1| > 1$$, or very small if <span class="math inline">$$|\lambda_1| < 1$$</span>. For this reason, we use **_normalized_** power iteration.

Normalized power iteration, is given by the following. Let $$\mathbf{x}_0$$ be a vector with unit norm: $$\|\mathbf{x}_0\| = 1$$ (any norm is fine), with $$\mathbf{x}_0 = \alpha_1 \mathbf{u}_1 + \alpha_2\mathbf{u}_2 + \dots \alpha_n\mathbf{u}_n,\text{ and }\alpha_1 \neq 0$$.

**_Normalized power iteration_** is defined by the following iterative sequence for $$k=1,2,3,\dots$$:

$$ \begin{align} &\mathbf{y}_k = \mathbf{A}\mathbf{x}_{k-1} \\ &\mathbf{x}_k = \frac{\mathbf{y}_k}{\|\mathbf{y}_k\|} \end{align} $$

where the norm $$\|\cdot\|$$ is identical to the norm used when we assumed $$\|\mathbf{x}_0\| = 1$$.

It can be shown that this sequence satisfies

$$\mathbf{x}_k = \frac{\mathbf{A}^k\mathbf{x}_0}{\|\mathbf{A}^k\mathbf{x}_0\|}. $$

This means that for large values of <span>$$k$$</span>, we have

$$\mathbf{x}_k \approx \left(\frac{\lambda_1}{|\lambda_1|}\right)^k\cdot\frac{\alpha_1\mathbf{u}_1}{\|\alpha_1\mathbf{u}_1\|}. $$

The largest eigenvalue could be positive, negative, or a complex number. In each case we will have:

$$
\begin{align} \lambda_1 > 0 \implies &\mathbf{x}_k \approx \frac{\alpha_1\mathbf{u}_1}{\|\alpha_1\mathbf{u}_1\|}\hspace{22.5mm} \mathbf{x}_k\text{ converges} \\ \lambda_1 < 0 \implies &\mathbf{x}_k \approx (-1)^k \frac{\alpha_1\mathbf{u}_1}{\|\alpha_1\mathbf{u}_1\|}\hspace{11.5mm} \text{in the limit, }\mathbf{x}_k\text{ alternates between }\pm\frac{\alpha_1\mathbf{u}_1}{\|\alpha_1\mathbf{u}_1\|}\\ \lambda_1 = re^{i\theta} \implies & \mathbf{x}_k \approx e^{ik\theta} \frac{\alpha_1\mathbf{u}_1}{\|\alpha_1\mathbf{u}_1\|} \hspace{16mm} \text{in the limit, }\mathbf{x}_k \text{ is a scalar multiple of } \mathbf{u}_1 \text{ with coefficient that rotates around the unit circle}.
\end{align} $$

Strictly speaking, normalized power iteration only converges to a single vector if $$\lambda_1 > 0$$, but $$\mathbf{x}_k$$ will be close to a scalar multiple of the eigenvector $$\mathbf{u}_1$$ for large values of <span>$$k$$</span>, regardless of whether the dominant eigenvalue is positive, negative, or complex. So normalized power iteration will work for any value of $$\lambda_1$$, as long as it is strictly bigger in magnitude than the other eigenvalues.

## Power Iteration code

The following code snippet performs power iteration:

```python
import numpy as np
def power_iter(A, x_0, p):
  # A: nxn matrix, x_0: initial guess, p: type of norm
  x_0 = x_0/np.linalg.norm(x_0,p)
  x_k = x_0
  for i in range(max_iterations):
    y_k = A @ x_k
    x_k = y_k/np.linalg.norm(y_k,p)
  return x_k
```

#### Example: Two Steps of Power Iteration

We'll use normalized power iteration (with the infinity norm) to approximate an eigenvector of the following matrix:
$$ \mathbf{A} = \begin{bmatrix} 1 & -2 \\ -1 & 1 \end{bmatrix}, $$
and the following initial guess:
$$ \mathbf{x}_0 = \begin{bmatrix} -1 \\ 0 \end{bmatrix} $$

**First Iteration**:

$$
\begin{align} &\mathbf{y}_1 = \mathbf{A}\mathbf{x}_0 = \begin{bmatrix} 1 & -2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} -1 \\ 0 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix},\\[15pt] &\mathbf{x}_1 = \frac{\mathbf{y}_1}{\|\mathbf{y}_1\|_{\infty}} = \mathbf{y}_1 = \begin{bmatrix} -1 \\ 1\end{bmatrix}. \end{align}
$$

**Second Iteration**:

$$
\begin{align} &\mathbf{y}_2 = \mathbf{A}\mathbf{x}_1 = \begin{bmatrix} 1 & -2 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} -3 \\ 2 \end{bmatrix},\\[15pt] &\mathbf{x}_2 = \frac{\mathbf{y}_2}{\|\mathbf{y}_2\|_{\infty}} = \frac{1}{3}\mathbf{y}_2= \begin{bmatrix} -1 \\ \frac{2}{3}\end{bmatrix} = \begin{bmatrix} -1 \\ 0.6666\dots\end{bmatrix}. \end{align}
$$

Even after only two iterations, we are getting close to a corresponding eigenvector:

$$\mathbf{u}_1 = \begin{bmatrix} -1 \\ \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} -1 \\ 0.7071 \end{bmatrix} $$

with relative error about 4 percent when measured in the infinity norm.

## Computing Eigenvalues from Eigenvectors

Power iteration allows us to find an approximate eigenvector corresponding to the largest eigenvalue in magnitude. How can we compute the actual eigenvalue from this? 

If $$\lambda$$ is an eigenvalue of $$\mathbf{A}$$, with corresponding eigenvector $$\mathbf{u}$$, then we can compute the value of $$\lambda$$ using the **_Rayleigh Quotient_**:

$$ \lambda = \frac{\mathbf{u}^T\mathbf{A}\mathbf{u}}{\mathbf{u}^T\mathbf{u}}. $$

Thus, one can compute an approximate eigenvalue using the approximate eigenvector found during power iteration.

## Power Iteration and Floating-Point Arithmetic

Recall that we made the assumption that the initial guess satisfies

$$\mathbf{x}_0 = \alpha_1 \mathbf{u}_1 + \alpha_2\mathbf{u}_2 + \dots \alpha_n\mathbf{u}_n,\text{ with }\alpha_1 \neq 0. $$

What happens if we choose an initial guess where $$\alpha_1 = 0$$? If we further assume that $$\vert\lambda_2\vert > \vert\lambda_3\vert\geq \vert\lambda_4\vert \geq \dots \geq\vert\lambda_n\vert$$, then in theory

$$ \mathbf{A}^k\boldsymbol{x}_0= \lambda_2^k \left(\alpha_2 {\bf u}_2 + \alpha_3\left(\frac{\lambda_3}{\lambda_2}\right)^k{\bf u}_3 + \dots + \alpha_n \left(\frac{\lambda_n}{\lambda_2}\right)^k {\bf u}_n\right), $$

and we would expect that

$$ \lim_{k\to\infty}\frac{\mathbf{A}^k \boldsymbol{x}_0}{\lambda_2^{k}} = \alpha_2 {\bf u}_2. $$

In practice, this does not happen. For one thing, choosing an initial guess such that $$\alpha_1 = 0$$ is extremely unlikely if we have no prior knowledge about the eigenvector $$\mathbf{u}_1$$. Since power iteration is performed numerically, using finite precision arithmetic, we will encounter the presence of rounding error in every iteration. This means that at every iteration $$k\text{, including }k = 0$$, we will instead have

$$ \mathbf{A}^k\hat{\boldsymbol{x}}_0= \lambda_1^k \left(\hat{\alpha}_1 \boldsymbol{u}_1 + \hat{\alpha}_2\left(\frac{\lambda_2}{\lambda_1}\right)^k\boldsymbol{u}_2 + \dots + \hat{\alpha}_n \left(\frac{\lambda_n}{\lambda_1}\right)^k \boldsymbol{u}_n\right), $$

where the $$\hat{\alpha}_k$$ are the approximate expansion coefficients of the rounded result. Even if $$\alpha_1 = 0$$, the finite precision representation $$\hat{\mathbf{x}}_0$$, will very likely have expansion coefficient $$\hat{\alpha}_1 \neq 0$$. Even in the case where rounding the initial guess does not introduce a non-zero $$\hat{\alpha}_1$$, rounding after applying the matrix $$\mathbf{A}$$ will almost certainly introduce a non-zero component in the dominant eigenvector after enough iterations. The probability of coming up with a starting guess $$\mathbf{x}_0$$ such that $$\hat{\alpha}_1 = 0$$ for all iterations is very, very low, if not impossible.

## Power Iteration without a Dominant Eigenvalue

Above, we assumed that one eigenvalue had magnitude strictly larger than all the others: $$\vert\lambda_1\vert > \vert\lambda_2\vert\geq \vert\lambda_3\vert \geq \dots \geq\vert\lambda_n\vert$$. What happens if $$\vert\lambda_1\vert = \vert\lambda_2\vert$$?

If $$\lambda_1 = \lambda_2 = \lambda \in \mathbb{R}$$, then:

$$ \mathbf{x}_k = \mathbf{A}^k\mathbf{x}_0 \approx \alpha_1\lambda^k\mathbf{u}_1 + \alpha_2\lambda^k\mathbf{u}_2 = \lambda^k\left(\alpha_1\mathbf{u}_1 + \alpha_2\mathbf{u}_2\right), $$

hence

$$\lim_{k\to\infty}\lambda^{-k}\mathbf{A}^k\mathbf{x}_0 = \alpha_1\mathbf{u}_1 + \alpha_2\mathbf{u}_2. $$

The quantity $$\alpha_1\mathbf{u}_1 + \alpha_2\mathbf{u}_2$$ is still an eigenvector corresponding to $$\lambda$$, so power iteration will still approach a dominant eigenvector.

If the dominant eigenvalues have opposite sign, i.e., $$\lambda_1 = -\lambda_2 = \lambda \in \mathbb{R}$$, then

 $$ \mathbf{x}_k = \mathbf{A}^k\mathbf{x}_0 \approx \alpha_1\lambda^k\mathbf{u}_1 + \alpha_2(-\lambda)^k\mathbf{u}_2 = \lambda^k\left(\alpha_1\mathbf{u}_1 + (-1)^k\alpha_2\mathbf{u}_2\right). $$

 For large <span>$$k$$</span>, we will have $$\lambda^{-k}\mathbf{A}\mathbf{x}_0 \approx \alpha_1\mathbf{u}_1 + (-1)^k\alpha_2\mathbf{u}_2$$, which although is a linear combination of two eigenvectors, is **_not_** itself an eigenvector of $$\mathbf{A}$$.

Finally, if the two dominant eigenvalues are a complex-conjugate pair $$\lambda_1 = re^{i\theta},\ \lambda_2 = re^{-i\theta}$$, then
$$ \mathbf{x}_k = \mathbf{A}^k\mathbf{x}_0 \approx \alpha_1\lambda^k\mathbf{u}_1 + \alpha_2(\overline{\lambda})^k\mathbf{u}_2 = \lambda^k\left(\alpha_1\mathbf{u}_1 + \left(\frac{\overline{\lambda}}{\lambda}\right)^k\alpha_2\mathbf{u}_2\right) = \lambda^k(\alpha_1\mathbf{u}_1 + \alpha_2 e^{-i2k\theta}\mathbf{u}_2). $$

For large <span>$$k$$</span>, $$\lambda^{-k}\mathbf{A}\mathbf{x}_0$$ approximate a linear combination of two eigenvectors, but this linear combination will not itself be an eigenvector.

## Inverse Iteration

To obtain an eigenvector corresponding to the **_smallest_** eigenvalue $$\lambda_n$$ of a non-singular matrix, we can apply power iteration to $$\mathbf{A}^{-1}$$. The following recurrence relationship describes inverse iteration algorithm:
$$\boldsymbol{x}_{k+1} = \frac{\mathbf{A}^{-1} \boldsymbol{x}_k}{\|\mathbf{A}^{-1} \boldsymbol{x}_k\|}.$$ Do not forget to nomalize each $$\boldsymbol{x}_{k+1}.$$

## Inverse Iteration with Shift

To obtain an eigenvector corresponding to the eigenvalue closest to some value $$\sigma$$, $$\mathbf{A}$$ can be shifted by $$\sigma$$ and inverted in order to solve it similarly to the power iteration algorithm. The following recurrence relationship describes inverse iteration algorithm:
$$\boldsymbol{x}_{k+1} = \frac{(\mathbf{A} - \sigma \mathbf{I})^{-1} \boldsymbol{x}_k}{\|(\mathbf{A} - \sigma \mathbf{I})^{-1} \boldsymbol{x}_k\|}$$. Note that this is identical to inverse iteration if the shift is zero.

## Rayleigh Quotient Iteration

The shift $$\sigma$$ can be updated based on a current estimate of the eigenvalue in order to improve convergence rate. Such an estimate can be found using the Rayleigh Quotient. Rayleigh Quotient Iteration is given by the following recurrence relation:

$$\sigma_k = \frac{\boldsymbol{x}_k^T \mathbf{A} \boldsymbol{x}_k}{\boldsymbol{x}_k^T\boldsymbol{x}_k}$$

$$\boldsymbol{x}_{k+1} = \frac{(\mathbf{A} - \sigma_k \mathbf{I})^{-1} \boldsymbol{x}_k}{\|(\mathbf{A} - \sigma_k \mathbf{I})^{-1} \boldsymbol{x}_k\|}.$$

## Convergence properties

The convergence rate for power iteration is **_linear_** and the recurrence relationship for the error between the current iterate and a dominant eigenvector is given by:
$$ \mathbf{e}_{k+1} \approx \frac{|\lambda_2|}{|\lambda_1|}\mathbf{e}_k$$.

The convergence rate for (shifted) inverse iteration is also linear, but now depends on the two closest eigenvalues to the shift $$\sigma$$. (Remember that standard inverse iteration corresponds to a shift $$\sigma = 0$$.) The recurrence relationship for the errors is given by:
$$ \mathbf{e}_{k+1} \approx \frac{|\lambda_\text{closest} - \sigma|}{|\lambda_\text{second-closest} - \sigma|}\mathbf{e}_k.$$

## Cost and Convergence Summary

| Method                         | Description                                                                             | Cost             | Convergence                                           |
|--------------------------------|-----------------------------------------------------------------------------------------|------------------|-------------------------------------------------------|
| Power Method                   | $$\boldsymbol{x}_{k+1} = \mathbf{A} \boldsymbol{x}_{k}$$                               | $$kn^2$$         | $$\left\|\frac{\lambda_2}{\lambda_1}\right\|$$       |
| Inverse Power Method           | $$\mathbf{A} \boldsymbol{x}_{k+1} = \boldsymbol{x}_{k}$$                                | $$n^{3} + kn^2$$ | $$\left\|\frac{\lambda_n}{\lambda_{n-1}}\right\|$$   |
| Shifted Inverse Power Method   | $$(\mathbf{A} - \sigma \mathbf{I}) \boldsymbol{x}_{k+1} = \boldsymbol{x}_{k}$$          | $$n^{3} + kn^2$$ | $$\left\|\frac{\lambda_c-\sigma}{\lambda_{c2}-\sigma}\right\|$$ |


$$\lambda_1$$: largest eigenvector (in magnitude) \\
$$\lambda_2$$: second largest eigenvector (in magnitude) \\
$$\lambda_n$$: smallest eigenvector (in magnitude) \\
$$\lambda_{n-1}$$: second smallest eigenvector (in magnitude) \\
$$\lambda_c$$: closest eigenvector to $$\sigma$$ \\
$$\lambda_{c2}$$: second closest eigenvector to $$\sigma$$

## Orthogonal Matrices

Square matrices are called **_orthogonal_** if and only if the columns are mutually orthogonal to one another and have a norm of <span>$$1$$</span> (such a set of vectors are formally known as an **_orthonormal_** set), i.e.:
$$\boldsymbol{c}_i^T \boldsymbol{c}_j = 0 \quad \forall \ i \neq j, \quad \|\boldsymbol{c}_i\| = 1 \quad \forall \ i \iff \mathbf{A} \in \mathcal{O}(n),$$
or
$$ \langle\boldsymbol{c}_i,\boldsymbol{c}_j \rangle = \begin{cases} 0 \quad \mathrm{if} \ i \neq j, \\ 1 \quad \mathrm{if} \ i = j \end{cases} \iff \mathbf{A} \in \mathcal{O}(n),$$
where $$\mathcal{O}(n)$$ is the set of all $$n \times n$$ orthogonal matrices called the orthogonal group, $$\boldsymbol{c}_i$$, $$i=1, \dots, n$$, are the columns of <span>$$\mathbf{A}$$</span>, and $$\langle \cdot, \cdot \rangle$$ is the inner product operator. Orthogonal matrices have many desirable properties:

(a) $$ \mathbf{A}^T \in \mathcal{O}(n) $$\\
(b) $$ \mathbf{A}^T \mathbf{A} = \mathbf{A} \mathbf{A}^T = \mathbf{I} \implies \mathbf{A}^{-1} = \mathbf{A}^T $$\\
(c) $$ \det{\mathbf{A}} = \pm 1 $$\\
(d) $$ \kappa_2(\mathbf{A}) = 1 $$

## Gram-Schmidt

The algorithm to construct an orthogonal basis from a set of linearly independent vectors is called the Gram-Schmidt process. For a basis set $$\{x_1, x_2, \dots x_n\}$$, we can form an orthogonal set $$\{v_1, v_2, \dots v_n\}$$ given by the following transformation:\\
$$\begin{align} \boldsymbol{v}_1 &= \boldsymbol{x}_1, \\ \boldsymbol{v}_2 &= \boldsymbol{x}_2 - \frac{\langle\boldsymbol{v}_1,\boldsymbol{x}_2\rangle}{\|\boldsymbol{v}_1\|^2}\boldsymbol{v}_1,\\ \boldsymbol{v}_3 &= \boldsymbol{x}_3 - \frac{\langle\boldsymbol{v}_1,\boldsymbol{x}_3\rangle}{\|\boldsymbol{v}_1\|^2}\boldsymbol{v}_1 - \frac{\langle\boldsymbol{v}_2,\boldsymbol{x}_3\rangle}{\|\boldsymbol{v}_2\|^2}\boldsymbol{v}_2,\\ \vdots &= \vdots\\ \boldsymbol{v}_n &= \boldsymbol{x}_n - \sum_{i=1}^{n-1}\frac{\langle\boldsymbol{v}_i,\boldsymbol{x}_n\rangle}{\|\boldsymbol{v}_i\|^2}\boldsymbol{v}_i, \end{align}$$\\
where $$\langle \cdot, \cdot \rangle$$ is the inner product operator. Each of the vectors in the new orthogonal set $$\{v_1, v_2, \dots v_n\}$$ can be **normalized** independently to obtain an **orthonormal basis**.

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-12-eigen.html)

## ChangeLog

* 2024-02-11 Pascal Adhikary <pascala2@illinois.edu>: add ev examples, cost table, reorganize
* 2022-02-28 Yuxuan Chen <yuxuan19@illinois.edu>: added learning objectives, cost summary
* 2020-03-01 Peter Sentz: added text to include content from slides
* 2018-10-14 Erin Carrier <ecarrie2@illinois.edu>: removes orthogonal/GS sections
* 2018-01-14 Erin Carrier <ecarrie2@illinois.edu>: removes demo links
* 2017-11-10 Erin Carrier <ecarrie2@illinois.edu>: adds costs of methods
* 2017-10-26 Matthew West <mwest@illinois.edu>: rewrote eval/evec definitions
* 2017-10-25 Erin Carrier <ecarrie2@illinois.edu>: minor fixes, added review questions
* 2017-10-14 Arun Lakshmanan <lakshma2@illinois.edu>: first complete draft
* 2017-10-16 Luke Olson <lukeo@illinois.edu>: outline
