---
title: Taylor Series
description: A representation of a function as an infinite sum of terms.
sort: 6

author:
  - CS 357 Course Staff
changelog:
  - 
    name: Dev Singh
    netid: dsingh14
    date: 2024-03-29
    message: added clarifications on taylor series error bound and approximations
  - 
    name: Arnav Aggarwal
    netid: arnava4
    date: 2024-03-27
    message: aligned notes with slides and added additional examples
  - 
    name: Arnav Shah
    netid: arnavss2
    date: 2022-01-25
    message: Added error predictions from slides
  -
    name: Mariana Silva
    netid: mfsilva
    date: 2021-01-20
    message: Removed FD content
  - 
    name: Peter Sentz
    netid: sentz2
    date: 2020-02-10
    message: Correct some small mistakes and update some notation
  - 
    name: John Doherty
    netid: jjdoher2
    date: 2019-01-29
    message: Added Finite Difference section from F18 activity
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-01-14
    message: removes demo links
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-02
    message: adds changelog
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-10-27
    message: adds review questions, minor fixes throughout
  - 
    name: Yu Meng
    netid: yumeng5
    date: 2017-10-27
    message: first complete draft
  - 
    name: Luke Olson
    netid: lukeo
    date: 2017-10-17
    message: outline
---

## Learning Objectives

- Approximate a function using a Taylor series
- Approximate function derivatives using a Taylor series
- Quantify the error in a Taylor series approximation

## Polynomial Overview

### Degree *n* Polynomial

A polynomial in a variable <span>\\(x\\)</span> can always be written (or rewritten) in the form

<div>\[ a_{n}x^{n}+a_{n-1}x^{n-1}+\dotsb +a_{2}x^{2}+a_{1}x+a_{0} \]</div>
where <span>\\(a_{i}\\)</span> (\\(0 \le i \le n\\)) are constants.

Using the summation notation, we can express the polynomial concisely by

<div>\[ \sum_{k=0}^{n} a_k x^k. \]</div>

If \\(a_n \neq 0\\), the polynomial is called an <span>\\(n\\)</span>-th degree polynomial.

### Degree *n* Polynomial as a Linear Combination of Monomials

A monomial in a variable <span>\\(x\\)</span> is a power of <span>\\(x\\)</span> where the exponent is a nonnegative integer (i.e. <span>\\(x^n\\)</span> where <span>\\(n\\)</span> is a nonnegative integer). You might see another definition of monomial which allows a nonzero constant as a coefficient in the monomial (i.e. <span>\\(a x^n\\)</span> where <span>\\(a\\)</span> is nonzero and <span>\\(n\\)</span> is a nonnegative integer). Then an <span>\\(n\\)</span>-th degree polynomial

<div>\[ \sum_{k=0}^{n} a_k x^k \]</div>
can be seen as a linear combination of monomials \\(\{x^i\ |\ 0 \le i \le n\}\\).


## Taylor Series Expansion

### Taylor Series Expansion: Infinite

A Taylor series is a representation of a function as an infinite sum of terms that are calculated from the values of the function's derivatives at a single point. The Taylor series expansion about <span>\\(x=x_0\\)</span> of a function <span>\\(f(x)\\)</span> that is infinitely differentiable at <span>\\(x_0\\)</span> is the power series

<div>\[ f(x_0)+\frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\frac{f'''(x_0)}{3!}(x-x_0)^3+\dotsb \]</div>

Using the summation notation, we can express the Taylor series concisely by

<div>\[ \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k .\]</div>
(Recall that <span>\\(0! = 1\\)</span>)

The infinite Taylor series expansion of any polynomial is the polynomial itself.

### Taylor Series Expansion: Finite

In practice, however, we often cannot compute the (infinite) Taylor series of the function, or the function is not infinitely differentiable at some points. Therefore, we often have to truncate the Taylor series (use a finite number of terms) to approximate the function.

### Taylor Series Approximation of Degree *n*

If we use the first <span>\\(n+1\\)</span> terms of the Taylor series, we will get

<div>\[ T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k ,\]</div>
which is called a Taylor polynomial of degree <span>\\(n\\)</span>.

The finite Taylor series expansion of degree <span>\\(n\\)</span> for any polynomial is the polynomial itself truncated to degree <span>\\(n\\)</span>.

### Examples

#### Example of a Taylor Series Expansion

How would we expand \\(f(x) = \cos x\\) about the point <span>\\(x_0 = 0\\)</span>, following the formula

$$
f(x) = f(x_0)+\frac{ f'(x_0) }{1!}(x-x_0)+\frac{ f''(x_0) }{2!}(x-x_0)^2+\frac{ f'''(x_0) }{3!}(x-x_0)^3+\dotsb
$$

<details>
    <summary><strong>Answer</strong></summary>

Well, we need to compute the derivatives of \(f(x)\) = cos \(x\) at <span>\(x = x_0\)</span>.

$$
\begin{align}
f(x_0) &= \cos(0) = 1\\
f'(x_0) &= -\sin(0) = 0\\
f''(x_0) &= -\cos(0) = -1\\
f'''(x_0) &= \sin(0) = 0\\
f^{(4)}(x_0) &= \cos(0) = 1\\
&\vdots \end{align}
$$

Then

$$
\begin{align}
\cos x &= f(0)+\frac{f'(0)}{1!}x+\frac{f''(0)}{2!}x^2+\frac{f'''(0)}{3!}x^3+\dotsb\\
&= 1 + 0 - \frac{1}{2}x^2 + 0 +\dotsb\\ &= \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k)!}x^{2k}
\end{align}
$$

</details>

#### Example of Using a Truncated Taylor Series to Approximate a Function

How would we approximate \\(f(x) = \sin x\\) at <span>\\(x = 2\\)</span> using a degree-4 Taylor polynomial about (centered at) the point <span>\\(x_0 = 0\\)</span>, following the formula

<div>\[ f(x) \approx f(x_0)+\frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\frac{f'''(x_0)}{3!}(x-x_0)^3+\frac{f^{(4)}(x_0)}{4!}(x-x_0)^4, \]</div>

<details>
    <summary><strong>Answer</strong></summary>

Well, we need to compute the first 4 derivatives of \(f(x)\) = sin \(x\) at <span>\(x = x_0\)</span>.

<div>\[\begin{align} f(x_0) &= \sin(0) = 0\\ f'(x_0) &= \cos(0) = 1\\ f''(x_0) &= -\sin(0) = 0\\ f'''(x_0) &= -\cos(0) = -1\\ f^{(4)}(x_0) &= \sin(0) = 0 \end{align}\]</div>
Then
<div>\[\begin{align} \sin x &\approx f(0)+\frac{f'(0)}{1!}x+\frac{f''(0)}{2!}x^2+\frac{f'''(0)}{3!}x^3+\frac{f^{(4)}(0)}{4!}x^4\\ &= 0 + x + 0 - \frac{1}{3!}x^3 + 0 \\ &= x - \frac{1}{6}x^3 \end{align}\]</div>

Using this truncated Taylor series centered at <span>\(x_0 = 0\)</span>, we can approximate \(f(x)\) = sin \(x\) at <span>\(x=2\)</span>. To do so, we simply plug <span>\(x = 2\)</span> into the above formula for the degree 4 Taylor polynomial giving

<div>\[\begin{align} \sin(2) &\approx 2 - \frac{1}{6} 2^3 \\ &\approx 2 - \frac{8}{6} \\ &\approx \frac{2}{3} \end{align}.\]</div>

We can always use Taylor polynomial with higher degrees to do the estimation. In a similar way that we derive the closed-form expression for cos \(x\), we can write the Taylor Series for sin \(x\) as
<div>\[ \sin x = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)!}x^{2k+1} \]</div>

</details>

## Taylor Series Error

### Error Bound when Truncating a Taylor Series

Suppose that <span>\\(f(x)\\)</span> is an <span>\\(n+1\\)</span> times differentiable function of <span>\\(x\\)</span>, and <span>\\(T_n(x)\\)</span> is the Taylor polynomial of degree <span>\\(n\\)</span> for <span>\\(f(x)\\)</span> centered at <span>\\(x_0\\)</span>. Then when \\(h = |x-x_0| \to 0\\), we obtain the truncation error bound by
\\[
\left|f(x)-T_n(x)\right|\le C \cdot h^{n+1} = O(h^{n+1})
\\]

We will see the exact expression of <span>\\(C\\)</span> in the next section: *Taylor Remainder Theorem*.

### Taylor Remainder Theorem

Suppose that <span>\\(f(x)\\)</span> is an <span>\\(n+1\\)</span> times differentiable function of <span>\\(x\\)</span>. Let <span>\\(R_n(x)\\)</span> denote the difference between <span>\\(f(x)\\)</span> and the Taylor polynomial of degree <span>\\(n\\)</span> for <span>\\(f(x)\\)</span> centered at <span>\\(x_0\\)</span>. Then

<div>\[ R_n(x) = f(x) - T_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} (x-x_0)^{n+1} \]</div>

for some \\(\xi\\) between <span>\\(x\\)</span> and <span>\\(x_0\\)</span>. Thus, the constant <span>\\(C\\)</span> mentioned above is

<div>\[ \max\limits_{\xi} \frac{\vert f^{(n+1)}(\xi)\vert }{(n+1)!}\]</div>.

Note that this value is equivalent to the next term of the Taylor series.

### Asymptotic behavior of the error

Let's say we have <span>\\(f(x)\\)</span> approximated using <span>\\(t_n(x)\\)</span>. Suppose the given interval is $$h_1$$ between $$x_0$$ and $$x$$ and the error associated with it is $$e_1$$. Let's say we have another interval $$h_2$$ and we need to find the error $$e_2$$ associated with it.

Using the formula $$e = O(h^{n+1})$$, we get

$$e_1 \propto {h_1}^{n+1}$$

$$e_2 \propto {h_2}^{n+1}$$

\\[ \frac{e_1}{e_2} = (\frac{h_1}{h_2})^{n+1} \\]

$$ e_2 = (\frac{h_2}{h_1})^{n+1}e_1 $$

### Examples

#### Example of an Error Bound

Suppose we want to approximate \\(f(x) = \sin x\\) using a degree-4 Taylor polynomial expanded about the point <span>\\(x_0 = 0\\)</span>. How would we compute the error bound for this approximation, following the Taylor Remainder Theorem:

<div>\[ R_4(x) = \frac{f^{(5)}(\xi)}{5!} (x-x_0)^{5} \]</div>
for some \\(\xi\\) between <span>\\(x_0\\)</span> and <span>\\(x\\)</span>.

<details>
    <summary><strong>Answer</strong></summary>

If we want to find the upper bound for the absolute error, we are looking for an upper bound for $$\vert f^{(5)}(\xi)\vert $$

Since \(f^{(5)}(x)\) = cos \(x\), we have \(|f^{(5)}(\xi)|\le \cos(0) \rightarrow |f^{(5)}(\xi)|\le 1\). Then
\[
|R_4(x)| = \left|\frac{f^{(5)}(\xi)}{5!} (x-x_0)^{5}\right| = \frac{|f^{(5)}(\xi)|}{5!} |x|^{5} \le \frac{1}{120} |x|^{5}
\]

</details>

#### Example of Error Predictions

Suppose you expand $$\sqrt{x - 10}$$ in a Taylor polynomial of degree 3 about the center $$x_0 = 12$$. For $$h_1 = 0.5$$, you find that the Taylor truncation error is about $$10^{-4}$$. How would you find the Taylor truncation error for $$h_2 = 0.25$$?

<details>
    <summary><strong>Answer</strong></summary>

We can use the formula \(e_2 = (\frac{h_2}{h_1})^{n+1}e_1\) to find the Taylor truncation error for \(h_2 = 0.25\).

<br>
<br>

Here, \(n = 3\) and hence \(e_2 = (\frac{0.25}{0.5})^{4} \cdot 10^{-4} = 0.625 \cdot 10^{-5}\).

</details>

## Review Questions

<ol>
  <li> What is the general form of a Taylor series?</li>
  <li> How do you use a Taylor series to approximate a function at a given point?</li>
  <li> How can you approximate the derivative of a function using Taylor series?</li>
  <li> How can you approximate the integral of a function using Taylor series?</li>
  <li> Given a function and a center, can you write out the <span><i>n</i></span>-th degree Taylor polynomial?</li>
  <li> For an <span><i>n</i></span>-th degree Taylor polynomial, what is the bound on the error of your approximation as a function of distance from the center?</li>
  <li> For simple functions, can you find the constant <span><i>C</i></span> in the Taylor error bound?</li>
  <li> Be able to determine how many terms are required for a Taylor series approximation to have less than some given error.</li>
  
</ol>