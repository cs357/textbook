---
title: Errors and Complexity
description: Different types of error, Big-O and complexities.
sort: 3
author: 
  - CS 357 Course Staff
changelog:
  -
    name: Kriti Chandak
    netid: kritic3
    date: 2024-03-30
    message: added information from slides and added examples
  - 
    name: Victor Zhao
    netid: chenyan4
    date: 2022-01-27
    message: fix terminology for accurate significant digits, fix the rule-of-thumb inequality
  -
    name: Victor Zhao
    netid: chenyan4
    date: 2022-01-20
    message: change notation for true value
  -
    name: Mariana Silva
    netid: mfsilva
    date: 2020-04-25
    message: small text revisions
  - 
    name: Peter Sentz
    netid: sentz2
    date: 2020-02-19
    message: add section on role of constants, change Big-Oh's to "mathcal"
  -
    name: Wanjun Jiang
    netid: wjiang24
    date: 2020-01-26
    message: add scientific notations, digits and figures
  - 
    name: Aming Ni
    netid: amingni2
    date: 2018-01-31
    message: changed three graphs
  -
    name: Yu Meng
    netid: yumeng5
    date: 2018-01-16
    message: minor fixes throughout
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-02
    message: adds changelog
  -
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-10-26
    message: adds review questions, minor changes throughout to better match termiology in class notes
  -
    name: John Doherty
    netid: jjdoher2
    date: 2017-10-23
    message: first complete draft
  - 
    name: Luke Olson
    netid: lukeo
    date: 2017-10-17
    message: outline
---

<br/><br/>

# Errors and Complexity
----

## Learning objectives

- Compare and contrast relative and absolute error
- Categorize a cost as \\(\mathcal{O}(n^p)\\)
- Categorize an error \\(\mathcal{O}(h^p)\\)
- Identify algebraic vs exponential growth and convergence

## Big Picture

- Numerical algorithms are distinguished by their **_cost_** and **_error_**, and the tradeoff between them.
- The algorithms or methods introduced in this course  indicate their error and cost whenever possible. These might be exact expressions or asymptotic bounds like \\(\mathcal{O}(h^2)\\) as \\(h \to 0\\) or \\(\mathcal{O}(n^3)\\) as \\(n \to \infty\\). For asymptotics we always indicate the limit.

## Scientific Notation

Using scientific notation, a number can be expressed in the form

$$
x = \pm r \times 10^{m}
$$

where \\(r\\) is a coefficient in the range \\(1 \geq r < 10 \\) and \\(m\\) is the exponent.

## Absolute and Relative Error

Results computed using numerical methods are inaccurate -- they
are approximations to the true values.  We can represent an
approximate result as a combination of the true value and some error:

$$\begin{eqnarray} \text{Approximate Result} = \text{True Value} + \text{Error} \end{eqnarray}$$


$$
\hat{x} = x + \Delta x
$$

Given this problem setup we can define the absolute error as:

$$
\begin{equation}
\text{Absolute Error} = |x - \hat{x}|
\end{equation}
.$$

This tells us how close our approximate result is to the actual answer.
However, absolute error can become an unsatisfactory and
misleading representation of the error depending on the magnitude of $$x$$.

<table class="table">
<colgroup>
<col width="50%" />
<col width="50%" />
</colgroup>
<thead>
<tr class="header">
<th style="color: black;">Case 1</th>
<th style="color: black;">Case 2</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><span class="math inline">\(x = 0.1\)</span>, <span class="math inline">\(\hat{x} = 0.2\)</span></td>
<td><span class="math inline">\(x = 100.0\)</span>, <span class="math inline">\(\hat{x} = 100.1\)</span></td>
</tr>
<tr class="even">
<td><span class="math inline">\(\mid x - \hat{x} \mid = 0.1\)</span></td>
<td><span class="math inline">\(\mid x - \hat{x} \mid = 0.1\)</span></td>
</tr>
</tbody>
</table>


In both of these cases, the absolute error is the same, 0.1.
However, we would intuitively consider case 2 more accurate than case 1
since our approximation is double the true value in case 1.
Because of this, we define the relative error, which will be an error estimate
independent of the magnitude. To obtain this we simply divide the absolute
error by the absolute value of the true value.

$$
\begin{equation}
\text{Relative Error} = \frac{|x - \hat{x}|}{|x|}
\end{equation}
$$

If we consider the two cases again, we can see that the relative error will be much lower in the second case, since relative error is independent of magnitude.

<table class="table">
<colgroup>
<col width="50%" />
<col width="50%" />
</colgroup>
<thead>
<tr class="header">
<th style="color: black;">Case 1</th>
<th style="color: black;">Case 2</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><span class="math inline">\(x = 0.1\)</span>, <span class="math inline">\(\hat{x} = 0.2\)</span></td>
<td><span class="math inline">\(x = 100.0\)</span>, <span class="math inline">\(\hat{x} = 100.1\)</span></td>
</tr>
<tr class="even">
<td><span class="math inline">\(\frac{\mid x - \hat{x} \mid}{\mid x \mid} = 1\)</span></td>
<td><span class="math inline">\(\frac{\mid x - \hat{x} \mid}{\mid x \mid} = 10^{-3}\)</span></td>
</tr>
</tbody>
</table>
## Significant Digits/Figures

**Significant figures** of a number are digits that carry meaningful information. They are digits beginning from the leftmost nonzero digit and ending with the rightmost "correct" digit, including final zeros that are exact. For example:

*   The number 3.14159 has six significant digits.
*   The number 0.00035 has two significant digits.
*   The number 0.000350 has three significant digits.
<br/>
The number of significant figures tells us about how many positions of $$x$$ and $$\hat{x}$$ agree.

An approximate result $$\hat{x}$$ has $$n$$ **significant figures** of a true value $$x$$ if the absolute error, $$\vert x - \hat{x}\vert$$, has zeros in the first $$n$$ decimal places counting from the leftmost nonzero (leading) digit of $$x$$, followed by a digit from 0 to 4.


**Example:** Assume $$x = 3.141592653$$ and suppose $$\hat{x}$$ is the approximate result:

$$
\hat{x} = 3.14159 \longrightarrow |x - \hat{x}| = 0.00000\mathbf{2}653 \longrightarrow \hat{x} \text{ has 6 significant figures.}
$$

$$
\hat{x} = 3.1415 \longrightarrow |x - \hat{x}| = 0.0000\mathbf{9}2653 \longrightarrow \hat{x} \text{ has 4 significant figures.}
$$

The number of accurate significant digits can be estimated by the relative error. If

$$
\text{Relative Error} = \frac{|x - \hat{x}|}{|x|} \geq 10^{-n + 1}
$$

then $$\hat{x}$$ has **at most** $$n$$ accurate significant digits. 

In general, we will use the rule-of-thumb for calculating an upper bound of the relative error: if an approximation has $$n$$ accurate significant digits, then the relative error is

$$
\frac{|x - \hat{x}|}{|x|} \leq 10^{-n+1}
$$



## Absolute and Relative Error of Vectors
If our calculated quantities are vectors then instead of using the absolute
value function, we can use the norm instead. Thus, our formulas become

$$
\begin{equation}
\text{Absolute Error} = \|\mathbf{x} - \mathbf{\hat{x}}\|
\end{equation}
$$

$$
\begin{equation}
\text{Relative Error} = \frac{\|\mathbf{x} - \mathbf{\hat{x}}\|}{\|\mathbf{x}\|}
\end{equation}
$$


We take the norm of the difference (and not the difference of the norms),
because we are interested in how far apart these two quantities are.
This formula is similar to finding that difference then using the vector
norm to find the length of that difference vector.


## Truncation Error vs. Rounding Error

**Rounding error** is the error that occurs from rounding values in a computation. This occurs constantly since computers use finite precision and have a limit on the memory available for storing one numerical value. Approximating $$\frac{1}{3} = 0.33333\dots$$ with a finite decimal expansion is an example of rounding error.


**Truncation error** is the error from using an approximate algorithm in place of an exact mathematical procedure or function. For example, in the case of evaluating functions, we may represent our function by a finite Taylor series up to degree $$n$$. The truncation error is the error that is incurred by not using the $$n+1$$ term and above.


## Big-O Notation

Big-O notation is used to understand and describe asymptotic behavior.
The definition in the cases of approching 0 or $$\infty$$ are as follows:


Let $$f$$ and $$g$$ be two functions. Then
$$f(x) = \mathcal{O}(g(x))$$ as $$x \rightarrow \infty$$
if and only if there exists a value $$M$$ and some $$x_0$$ such that
$$|f(x)| \leq M|g(x)|$$ $$\forall x$$ where $$x\geq x_0$$


Let $$f$$ and $$g$$ be two functions. Then
$$f(h) = \mathcal{O}(g(h))$$ as $$h \rightarrow 0$$
if and only if there exists a value $$M$$ and some $$h_0$$ such that
$$|f(h)| \leq M|g(h)|$$ $$\forall h$$ where $$0 < h < h_0$$


But what if we want to consider the function approaching an arbitrary value?
Then we can redefine the expression as:


Let $$f$$ and $$g$$ be two functions. Then
$$f(x) = \mathcal{O}(g(x))$$ as $$x \rightarrow a$$
if and only if there exists a value $$M$$ and some $$\delta$$ such that
$$|f(x)| \leq M|g(x)|$$ $$\forall x$$ where $$0 < |x − a| < \delta$$

For example, consider the function $$f(x) = 2x^2 + 27x + 1000$$

When $$x \rightarrow \infty$$, the term $$x^2$$ is the most significant, and hence $$f(x) = O(x^2)$$

## Big-O Examples - Time Complexity

We can use Big-O to describe the time complexity of our algorithms.

Consider the case of matrix-matrix multiplication.
If the size of each of our matrices is $$n \times n$$,
then the time it will take to multiply the matrices is $$\mathcal{O}(n^3)$$ meaning
that $$\text{Run time} \approx C \cdot n^3$$.  Suppose we know that for $$n_1=1000$$,
the matrix-matrix multiplication takes 5 seconds.  Estimate how much time
it would take if we double the size of our matrices to $$2n \times 2n$$.

We know that:


$$\begin{align*}
\text{Time}(2n_1) &\approx C \cdot (2n_1)^3 \\
&= C \cdot 2^3 \cdot n_1^3\\
&= 8 \cdot (C \cdot n_1^3) \\
&= 8 \cdot \text{Time}(n_1) \\
&= 40 \text{ seconds}
\end{align*}$$

So, when we double the size of our our matrices to
$$2n \times 2n$$, the time becomes $$(2n)^3 = 8n^3$$.
Thus, the runtime will be roughly 8 times as long.

## Big-O Examples - Truncation Errors

We can also use Big-O notation to describe the truncation error. A numerical method is called $$n$$-th order accurate if its
truncation error $$E(h)$$ obeys $$E(h) = \mathcal{O}(h^n)$$.

Consider solving an interpolation problem. We have an interval of
length $$h$$ where our interpolant is valid and we know that our approximation
is order $$\mathcal{O}(h^2)$$. What this means is that as we decrease h (the interval
length), our error will decrease quadratically. Using the definition of Big-O,
we know that $$\text{Error} = C \cdot h^2$$ where $$C$$ is some constant.

In some cases, we may not know the exponent in $$E(h) = \mathcal{O}(h^n)$$.  We can estimate it using by computing the error at two different values of $$h$$.  Suppose we have two quantities, $$h_1 = 0.5$$ and $$h_2 = 0.25$$.  We compute the corresponding errors as $$E(h_1) = 0.125$$ and $$E(h_2) = 0.015625$$.  Then, since $$E(h) = \mathcal{O}(h^n)$$, we have:


$$
\begin{eqnarray}\frac{0.125}{0.015625}  &=\frac{E(h_1)}{E(h_2)} \\
&\approx\frac{Ch_1^n}{Ch_2^n}\\
&=\left(\frac{h_1}{h_2}\right)^n\\
\end{eqnarray}
$$

$$
\begin{eqnarray}
\implies\log\left(\frac{0.125}{0.015625}\right)=
n\log\left(\frac{h_1}{h_2}\right) =n\log\left(\frac{0.5}{0.25}\right)
\end{eqnarray}
$$

Solving this equation for $$n$$, we obtain $$n = 3$$.


## Big-O Example - Role of Constants

It is important that one does not place too much importance on the constant $$M$$ in the definition of Big-O notation; it is essentially arbitrary.

Suppose $$f_1(n) = 10^{-20}n^2$$ and $$f_2(n) = 10^{20}n^2$$. While $$f_2$$ is much larger than $$f_1$$ for all values of $$n$$, **both** are $$\mathcal{O}(n^2)$$; this is obvious if we choose any constants $$M_1 \geq 10^{-20}$$ and $$M_2 \geq 10^{20}$$.

However, it is also true that $$f_2(n) = \mathcal{O}(10^{-20}n^2)$$ for any constant $$M \geq 10^{40}$$

$$\begin{eqnarray}
f_2(n) = 10^{20}n^2 = 10^{40} \times 10^{-20}n^2 \leq M\times 10^{-20}n^2.
\end{eqnarray}$$


So including a constant inside the $$\mathcal{O}$$ is basically meaningless.


**Question:** What is the function $$g(n)$$ that gives the tightest bound on $$f_2(n) = \mathcal{O}(g(n))$$?

**Solution:** the answer is $$g(n) = n^2$$. For any $$r < 2$$, there is **no** constant $$M$$ such that $$|f_2(n)| \leq Mn^r$$ for all
$$n$$ sufficiently large. So $$n^r$$ for $$r < 2$$ is not a bound on $$f_2$$. For any $$q > 2$$, there exist a pair of constants $$M_1$$ and $$M_2$$ such that for all $$n$$ sufficiently large:

$$\begin{align*} f_2(n) \leq M_1 n^2\leq M_2 n^q. \end{align*}$$

However, we **cannot** find a pair of constants $$M_3$$ and $$M_4$$ such that:

$$\begin{align*} f_2(n) \leq M_3 n^q\leq M_4 n^2. \end{align*}$$

Thus, we cannot "fit" another function in between $$f_2(n)$$ and $$n^2$$, so $$n^2$$ is the tightest bound.

One may be tempted to think the correct answer should actually be $$g(n) = 10^{20}n^2$$; however, this does not actually provide any additional information about the growth of $$f_2$$. Notice that we didn't specify what $$M_1$$ and $$M_2$$ were in the inequality above. Big-O notation says **nothing** about the size of the constant. The statements

$$\begin{align*} f_2(n) &= \mathcal{O}(n^2),\\ f_2(n) &= \mathcal{O}(10^{20}n^2),\\ f_2(n) &= \mathcal{O}(10^{-20}n^2), \end{align*}$$

are all equivalent, in that they all give the same amount of information on the growth of $$f_2$$, since the constants are not specified.  Since $$10^{-20}$$ is very small, it may be tempting to conclude that it is "tighter" than the other two, which is not true.  Therefore, it is always best practice to avoid placing unnecessary constants inside the $$\mathcal{O}$$, and we expect you do refrain from doing so in this course.


## Convergence Definitions

Algebraic growth/convergence is when the coefficients
$$a_n$$ in the sequence we are interested in
behave like $$\mathcal{O}(n^{\alpha})$$ for growth and $$\mathcal{O}(1/n^{\alpha})$$ for convergence,
where $$\alpha$$ is called the algebraic index of convergence.
A sequence that grows or converges
algebraically is a straight line in a log-log plot.

Exponential growth/convergence is when the coefficients $$a_n$$ of the sequence
we are interested in behave like $$\mathcal{O}(e^{qn^{\beta}})$$ for growth and
$$\mathcal{O}(e^{-qn^{\beta}})$$ for convergence,
where $$q$$ is a constant for some
$$\beta > 0$$. Exponential
growth is much faster than algebraic growth.  Exponential growth/convergence
is also sometimes called spectral growth/convergence. A sequence
that grows exponentially is a straight line in a log-linear plot.
Exponential convergence is often further classified as
supergeometric, geometric, or subgeometric convergence.


<div class="row">
<div class="col-lg"> <img src="{{ site.baseurl }}/assets/img/figs/convergence1.png" alt="Convergence" width="500" /> </div>
<div class="col-lg"> <img src="{{ site.baseurl }}/assets/img/figs/convergence2.png" alt="Convergence" width="500" /> </div>
<div class="col-lg"> <img src="{{ site.baseurl }}/assets/img/figs/convergence3.png" alt="Convergence" width="500" /> </div>
</div>

<p style="text-align: center;">Figures from  J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, 2nd ed., Dover, New
York, 2001.</p>

## Review Questions

1. What is the formula for relative error?

2. What is the formula for absolute error?

3. Which is usually the better way to calculate error?

4. If you have \\(k\\) accurate significant digits, what is the tightest bound on your relative error?

5. Given a maximum relative error, what is the largest (or smallest) value you could have?

6. How do you compute relative and absolute error for vectors?

7. What is truncation error?

8. What is rounding error?

9. What does it mean for an algorithm to take \\(O(n^3)\\) time?

10. Can you give an example of an operation that takes \\(O(n^2)\\) time?

11. What does it mean for error to follow \\(O(h^4)\\)?

12. Can you simplify \\(O(h^5 + h^2 + h)\\)? What assumption do you need to make?

13. If you know an operation is \\(O(n^4)\\), can you predict its time/error on unseen inputs from inputs that you already have?

14. If you are given runtime or error data for an operation, can you find the best \\(x\\) such that the operation is \\(O(n^x)\\)?

15. What is algebraic growth/convergence?

16. What is exponential growth/convergence?


## Links to other resources

- [Big-O Notation](https://www.stat.rice.edu/~dobelman/notes_papers/math/big_O.little_o.pdf)