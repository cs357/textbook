---
title: Condition Numbers
description: A way to measure how good a matrix is.
sort: 11
---
# Condition Numbers

* * *

## Learning Objectives

*   Compute the condition number
*   Quantify the impact of a high condition number

## Numerical experiments

**Input** has uncertainties:

* Errors due to representation with finite precision
* Error in the sampling

Once you select your numerical method, how much error should you expect to see in your output?

_Is your method sensitive to errors (perturbation) in the input?_


## Sensitivity of Solutions of Linear Systems and Error Bound

Suppose we start with a non-singular system of linear equations $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$. We change the right-hand side vector $$\boldsymbol{b}$$ (the input) by a small amount $$\Delta \boldsymbol{b}$$. How much does the solution $$\boldsymbol{x}$$ (the output) change, i.e., how large is $$\Delta \boldsymbol{x}$$. 


Let $$\boldsymbol{x}$$ be the solution of $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$ and $$\hat{\boldsymbol{x}}$$ be the solution of the perturbed problem $${\bf A} \hat{\boldsymbol{x}} = \boldsymbol{b} + \Delta \boldsymbol{b}$$. Let $$\Delta \boldsymbol{x} = \hat{\boldsymbol{x}} - \boldsymbol{x}$$ be the absolute error in output. Then we have
$${\bf A} \boldsymbol{x} + {\bf A} \Delta \boldsymbol{x} = \boldsymbol{b} + \Delta \boldsymbol{b}, $$
so
$${\bf A} \Delta \boldsymbol{x} = \Delta \boldsymbol{b}. $$
Now we want to see how the relative error in output $$\left(\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|}\right)$$ is related to the relative error in input $$\left(\frac{\|\Delta \boldsymbol{b}\|}{\|\boldsymbol{b}\|}\right)$$:

$$
\begin{align}
\frac{\|\Delta \boldsymbol{x}\| / \|\boldsymbol{x}\|}{\|\Delta \boldsymbol{b}\| / \|\boldsymbol{b}\|} &= \frac{\|\Delta \boldsymbol{x}\| \|\boldsymbol{b}\|}{\|\boldsymbol{x}\| \|\Delta \boldsymbol{b}\|}\\
&= \frac{\|{\bf A}^{-1} \Delta \boldsymbol{b}\| \|{\bf A} \boldsymbol{x}\|}{\|\boldsymbol{x}\| \|\Delta \boldsymbol{b}\|}\\
&\le \frac{\|{\bf A}^{-1}\| \|\Delta \boldsymbol{b}\| \|{\bf A}\| \|\boldsymbol{x}\|}{\|\boldsymbol{x}\| \|\Delta \boldsymbol{b}\|} \\
&= \|{\bf A}^{-1}\| \|{\bf A}\|\\ &= \text{cond}({\bf A})
\end{align}
$$

where we used $$\|{\bf A}\boldsymbol{x}\| \le \|{\bf A}\| \|\boldsymbol{x}\|, \forall \boldsymbol{x}.$$

Then

$$\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|} \le \text{cond}({\bf A})\frac{\|\Delta \boldsymbol{b}\|}{\|\boldsymbol{b}\|}  \qquad (1)$$ 

<br>


Instead of changing the input $$\boldsymbol{b}$$, we can also add a perturbation to the matrix $$\boldsymbol{A}$$ (input) by a small amount $$\boldsymbol{E}$$, such that 

$$
\begin{align}
(\boldsymbol{A} + \boldsymbol{E}) \hat{\boldsymbol{x}} = \boldsymbol{b}
\end{align}
$$

and in a similar way obtain:

$$\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|} \le \text{cond}({\bf A})\frac{\| \boldsymbol{E}\|}{\|\boldsymbol{A}\|}  $$


Therefore, if we know the relative error in input, then we can use the condition number of the system to obtain an upper bound for the relative error of our computed solution (output).


#### Examples

PLACEHOLDER

## Condition Number

#### Condition Number Definition

**_Definition_**: a measure of sensitivity of solving a linear system of equations to variations in the input.

The **_condition number_** of a square nonsingular matrix <span>$${\bf A}$$</span> is defined by
$$\text{cond}({\bf A}) = \kappa({\bf A}) = \|{\bf A}\| \|{\bf A}^{-1}\| $$
which is also the condition number associated with solving the linear system $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$. A matrix with a large condition number is said to be **_ill-conditioned_**, while a matrix with a small condition is said to be **_well-conditioned_**.

Recall that the induced matrix norm is given by:

$$
\begin{align}
\|{\bf A}\|_p := \max_{\|\mathbf{x}\|=1} \|{\bf A}\mathbf{x}\|_p
\end{align}
$$

The condition number can be measured with any <span>$$p$$</span>-norm, so to be precise we typically specify the norm being used, i.e. $$\text{cond}_2$$, $$\text{cond}_1, \text{cond}_{\infty}$$.

If <span>$${\bf A}$$</span> is singular ($${\bf A}^{-1}$$ does not exist), we can define $$\text{cond}({\bf A}) = \infty$$ by convention.

The identity matrix is well conditioned. Assuming the inverse of $$\|{\bf A}\|$$ exists, $$\text{cond}({\bf A}) = \|{\bf A}\| \|{\bf A}^{-1}\| \geq \|{\bf A}{\bf A}^{-1}\| = \|{\bf I}\| = 1.$$ This is the smallest possible condition number. Small condition numbers correspond to little error amplification. Remember, small condition numbers are good!


#### Recall Induced Matrix Norms

For the 1-norm, we take the maximum absolute column sum of the matrix $$\boldsymbol{A}$$

$$\|{\bf A}\|_1 = \max_j \sum_{i=1}^n \vert a_{ij} \vert.$$

For the $$\infty$$-norm, we take the maximum absolute row sum of the matrix $$\boldsymbol{A}$$

$$\|{\bf A}\|_{\infty} = \max_i \sum_{j=1}^n \vert a_{ij} \vert.$$

For the 2-norm, $$\sigma_k$$ are the singular values of the matrix $$\boldsymbol{A}$$

$$\|{\bf A}\|_{2} = \max_k \sigma_k$$

#### Condition Number of Orthogonal Matrices

What is the 2-norm condition number of an orthogonal matrix $$\boldsymbol{A}$$?

$$
\begin{align}
\text{cond}({\bf A}) = \|{\bf A}\|_2 \|{\bf A}^{-1}\|_2 = \|{\bf A}\|_2 \|{\bf A}^{T}\|_2 = 1
\end{align}
$$

Hence, this means that orthogonal matrices have optimal conditioning.

#### Things to Remember About Condition Numbers
*   For any matrix $${\bf A}$$, $$\text{cond}({\bf A}) \geq 1.$$
*   For the identity matrix $${\bf I}$$, $$\text{cond}({\bf I}) = 1.$$
*   For any matrix $${\bf A}$$ and a nonzero scalar $$\gamma$$, $$\text{cond}(\gamma {\bf A}) = \text{cond}({\bf A}).$$
*   For any diagonal matrix $${\bf D}$$, $$\text{cond}({\bf D}) $$ = $$\frac{\text{max}\mid d_{i} \mid}{\text{min}\mid d_{i} \mid}.$$
*   The condition number is a measure of how close a matrix is to being singular: a matrix with large condition number is nearly singular, whereas a matrix with a condition number close to 1 is far from being singular.
*   The determinant of a matrix is **NOT** a good indicator to check whether a matrix is near singularity.


#### Examples

PLACEHOLDER

## Residual vs Error

The **_residual vector_** $$\boldsymbol{r}$$ of approximate solution $$\hat{\boldsymbol{x}}$$ for the linear system $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$ is defined as
$$\boldsymbol{r} = \boldsymbol{b} - {\bf A} \hat{\boldsymbol{x}} $$. Since $${\bf A} \hat{\boldsymbol{x}} = \boldsymbol{b} + \Delta \boldsymbol{b}$$, we have

$$\boldsymbol{r} = \boldsymbol{b} - (\boldsymbol{b} + \Delta \boldsymbol{b}) = -\Delta \boldsymbol{b} $$

Therefore, equation (1) can also be written as

$$\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|} \le \text{cond}({\bf A})\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{b}\|} $$

If we define relative residual as $$\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{b}\|}$$, we can see that a small relative residual implies small relative error in approximate solution only if <span>$${\bf A}$$</span> is well-conditioned ($$\text{cond}({\bf A})$$ is small).

In addition, it's important to note the difference between relative residual and relative error. The relative residual $$\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{b}\|}$$ tells us how well the approximate solution $$\hat{\boldsymbol{x}}$$ satisfies the linear system $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$. The relative error $$\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|} $$ tells us how close the approximated solution $$\hat{\boldsymbol{x}}$$ is to the exact solution $$ \boldsymbol{x}$$. Keep in mind that we don't know the exact solution $$ \boldsymbol{x}$$, this is why we started using the residual vector $$ \boldsymbol{r}$$. 

#### Examples

PLACEHOLDER

## Alternative Definitions of Relative Residual
There are other closely related quantities that also have the name "relative residual".  Note that

$$\begin{align} \|\Delta \boldsymbol{x}\| &= \|\hat{\boldsymbol{x}} - \boldsymbol{x}\| \\
&= \|\boldsymbol{A}^{-1}\boldsymbol{A}\hat{\boldsymbol{x}} - \boldsymbol{A}^{-1}\boldsymbol{b}\| \\
&= \|\boldsymbol{A}^{-1}(\boldsymbol{A}\hat{\boldsymbol{x}} - \boldsymbol{b})\| \\
&= \|\boldsymbol{A}^{-1}\boldsymbol{r}\|\\
&\leq \|\boldsymbol{A}^{-1}\|\cdot \| \boldsymbol{r}\| \\
&= \|\boldsymbol{A}^{-1}\|\cdot \|\boldsymbol{A}\| \frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|} = \text{cond}(\boldsymbol{A})\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|}.
\end{align}$$

In summary,

$$\|\Delta \boldsymbol{x}\| \leq \text{cond}(\boldsymbol{A})\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|}\qquad (2)$$

We can divide this inequality by $$\|\boldsymbol{x}\|$$ to obtain

$$\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|} \le \text{cond}({\bf A})\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|\cdot\|\boldsymbol{x}\|}.$$

The quantity $$\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|\cdot\|\boldsymbol{x}\|}$$ is also known as the relative residual.  This inequality is useful mathematically, but involves the norm $$\|\mathbf{x}\|$$ of the unknown solution, so it isn't a practical way to bound the relative error.  Since $$\|\boldsymbol{b}\| = \|\boldsymbol{A}\boldsymbol{x}\| \leq \|\boldsymbol{A}\|\cdot \|\boldsymbol{x}\|$$, we have

$$\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|\cdot\|\boldsymbol{x}\|} \leq  \frac{\|\boldsymbol{r}\|}{\|\boldsymbol{b}\|}$$

but are sometimes equal for certain choices of $$\boldsymbol{b}$$.

We can also divide equation (2) by $$\|\hat{\boldsymbol{x}}\|$$ to obtain 

$$\frac{\|\Delta \boldsymbol{x}\|}{\|\hat{\boldsymbol{x}}\|} \le \text{cond}({\bf A})\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|\cdot\|\hat{\boldsymbol{x}}\|}.$$

The left-hand side is no longer the relative error (the denominator is the norm of the approximate solution now, not the exact solution), but the right-hand side can still provide a reasonable estimate of the relative error.  It is also computable, since the norm of the true solution does not appear on the right-hand side.

For this reason, the quantity $$\frac{\|\boldsymbol{r}\|}{\|\boldsymbol{A}\|\cdot\|\hat{\boldsymbol{x}}\|}$$ is also known as the relative residual.  This is used in the next section to describe the relationship between the residual and errors in the matrix $$\boldsymbol{A}$$.

While 3 different quantities all being named "relative residual" may be confusing, you should be able to determine which one is being discussed by the context.


## Gaussian Elimination (with Partial Pivoting) is Guaranteed to Produce a Small Residual

When we use Gaussian elimination with partial pivoting to compute the solution for the linear system
$${\bf A} \boldsymbol{x} = \boldsymbol{b}$$ and obtain an approximate solution $$\hat{\boldsymbol{x}}$$, the residual vector $$\boldsymbol{r}$$ satisfies:

$$\frac{\|\boldsymbol{r}\|}{\|{\bf A}\| \|\hat{\boldsymbol{x}}\|} \le \frac{\|E\|}{\|{\bf A}\|} \le c \epsilon_{mach} $$

where <span>$$E$$</span> is backward error in <span>$${\bf A}$$</span> (which is defined by $$({\bf A}+E)\hat{\boldsymbol{x}} = \boldsymbol{b}$$), <span>$$c$$</span> is a coefficient related to <span>$${\bf A}$$</span> and $$\epsilon_{mach}$$ is machine epsilon.

Typically <span>$$c$$</span> is small with partial pivoting, but <span>$$c$$</span> can be arbitrarily large without pivoting.

Therefore, Gaussian elimination with partial pivoting yields <b> small relative residual regardless of conditioning of the system <b>.


#### Examples

PLACEHOLDER

For more details, see [Gaussian Elimination & Roundoff Error](https://math.la.asu.edu/~gardner/lu-round.pdf).

## Accuracy Rule of Thumb for conditioning

Suppose we apply Gaussian elimination with partial pivoting and back substitution to the linear system $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$ and obtain a computed solution $$\hat{\boldsymbol{x}}$$. If the entries in <span>$${\bf A}$$</span> and $$\boldsymbol{b}$$ are accurate to <span>$$s$$</span> decimal digits, and $$\text{cond}({\bf A}) \approx 10^w$$, then the elements of the solution vector $$\hat{\boldsymbol{x}}$$ will be accurate to about <span>$$s-w$$</span> decimal digits.

For a proof of this rule of thumb, please see [Fundamentals of Matrix Computations by David S. Watkins](https://books.google.com/books?id=xi5omWiQ-3kC&pg=PA165&lpg=PA165&dq=gaussian+elimination+rule+of+thumb&source=bl&ots=KlQVax3zja&sig=o4SHiYPAXodkk39u9yw0NYZe1Zo&hl=en&sa=X&ved=0ahUKEwiopPykkvjWAhWjzIMKHYGpDIsQ6AEIXzAK#v=onepage&q=gaussian%20elimination%20rule%20of%20thumb&f=false).

#### Example
How many accurate decimal digits in the solution can we expect to obtain if we solve a linear system $${\bf A} \boldsymbol{x} = \boldsymbol{b}$$ where $$\text{cond}({\bf A}) = 10^{10}$$ using Gaussian elimination with partial pivoting, assuming we are using IEEE double precision and the inputs are accurate to machine precision?

In IEEE double precision, $$\epsilon_{mach} \approx 2.2\times 10^{-16}$$, which means the entries in $${\bf A}$$ and $$\boldsymbol{b}$$ are accurate to $$\vert\log_{10}(2.2\times 10^{-16})\vert \approx 16$$ decimal digits.

Then, using the rule of thumb, we know the entries in $$\hat{\boldsymbol{x}}$$ will be accurate to about <span>$$16-10 = 6$$</span> decimal digits.

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-10-condition.html)

## ChangeLog

*   2022-03-19 Yuxuan Chen <yuxuan19@illinois.edu>: added condition number bullet pounts, minor fixes
*   2017-10-27 Erin Carrier <ecarrie2@illinois.edu>: adds review questions, minor fixes throughout, revised rule of thumb wording
*   2017-10-27 Yu Meng <yumeng5@illinois.edu: first complete draft
*   2017-10-17 Luke Olson <lukeo.illinois.edu: outline
