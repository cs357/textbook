---
title: Taylor Series
description: A representation of a function as an infinite sum of terms.
sort: 6
---

# Taylor Series

---

## Learning Objectives

- Approximate a function using a Taylor series
- Approximate function derivatives using a Taylor series
- Quantify the error in a Taylor series approximation

## Degree <span>\\(n\\)</span> Polynomial

A polynomial in a variable <span>\\(x\\)</span> can always be written (or rewritten) in the form

<div>\[ a_{n}x^{n}+a_{n-1}x^{n-1}+\dotsb +a_{2}x^{2}+a_{1}x+a_{0} \]</div>
where <span>\\(a_{i}\\)</span> (\\(0 \le i \le n\\)) are constants.

Using the summation notation, we can express the polynomial concisely by

<div>\[ \sum_{k=0}^{n} a_k x^k. \]</div>

If \\(a_n \neq 0\\), the polynomial is called an <span>\\(n\\)</span>-th degree polynomial.

## Degree <span>\\(n\\)</span> Polynomial as a Linear Combination of Monomials

A monomial in a variable <span>\\(x\\)</span> is a power of <span>\\(x\\)</span> where the exponent is a nonnegative integer (i.e. <span>\\(x^n\\)</span> where <span>\\(n\\)</span> is a nonnegative integer). You might see another definition of monomial which allows a nonzero constant as a coefficient in the monomial (i.e. <span>\\(a x^n\\)</span> where <span>\\(a\\)</span> is nonzero and <span>\\(n\\)</span> is a nonnegative integer). Then an <span>\\(n\\)</span>-th degree polynomial

<div>\[ \sum_{k=0}^{n} a_k x^k \]</div>
can be seen as a linear combination of monomials \\(\{x^i\ |\ 0 \le i \le n\}\\).

## Taylor Series Expansion, Infinite

A Taylor series is a representation of a function as an infinite sum of terms that are calculated from the values of the function's derivatives at a single point. The Taylor series expansion about <span>\\(x=x_0\\)</span> of a function <span>\\(f(x)\\)</span> that is infinitely differentiable at <span>\\(x_0\\)</span> is the power series

<div>\[ f(x_0)+\frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\frac{f'''(x_0)}{3!}(x-x_0)^3+\dotsb \]</div>

Using the summation notation, we can express the Taylor series concisely by

<div>\[ \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k .\]</div>
(Recall that <span>\\(0! = 1\\)</span>)

## Taylor Series Expansion, Finite

In practice, however, we often cannot compute the (infinite) Taylor series of the function, or the function is not infinitely differentiable at some points. Therefore, we often have to truncate the Taylor series (use a finite number of terms) to approximate the function.

## Taylor Series Approximation of Degree <span>\\(n\\)</span>

If we use the first <span>\\(n+1\\)</span> terms of the Taylor series, we will get

<div>\[ T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k ,\]</div>
which is called a Taylor polynomial of degree <span>\\(n\\)</span>.

## Error Bound when Truncating a Taylor Series

Suppose that <span>\\(f(x)\\)</span> is an <span>\\(n+1\\)</span> times differentiable function of <span>\\(x\\)</span>, and <span>\\(T_n(x)\\)</span> is the Taylor polynomial of degree <span>\\(n\\)</span> for <span>\\(f(x)\\)</span> centered at <span>\\(x_0\\)</span>. Then when \\(h = |x-x_0| \to 0\\), we obtain the truncation error bound by
\\[
\left|f(x)-T_n(x)\right|\le C \cdot h^{n+1} = O(h^{n+1})
\\]

We will see the exact expression of <span>\\(C\\)</span> in the next section: Taylor Remainder Theorem.

## Taylor Remainder Theorem

Suppose that <span>\\(f(x)\\)</span> is an <span>\\(n+1\\)</span> times differentiable function of <span>\\(x\\)</span>. Let <span>\\(R_n(x)\\)</span> denote the difference between <span>\\(f(x)\\)</span> and the Taylor polynomial of degree <span>\\(n\\)</span> for <span>\\(f(x)\\)</span> centered at <span>\\(x_0\\)</span>. Then

<div>\[ R_n(x) = f(x) - T_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} (x-x_0)^{n+1} \]</div>

for some \\(\xi\\) between <span>\\(x\\)</span> and <span>\\(x_0\\)</span>. Thus, the constant <span>\\(C\\)</span> mentioned above is

<div>\[ \max\limits_{\xi} \frac{\vert f^{(n+1)}(\xi)\vert }{(n+1)!}\]</div>.

## Asymptotic behavior of the error

Let's say we have <span>\\(f(x)\\)</span> approximated using <span>\\(t_n(x)\\)</span>. Suppose the given interval is $$h_1$$ between $$x_0$$ and $$x$$ and the error associated with it is $$e_1$$. Let's say we have another interval $$h_2$$ and we need to find the error $$e_2$$ associated with it.

Using the formula $$e = O(h^{n+1})$$, we get

$$e_1 \propto {h_1}^{n+1}$$

$$e_2 \propto {h_2}^{n+1}$$

\\[ \frac{e_1}{e_2} = (\frac{h_1}{h_2})^{n+1} \\]

$$ e_2 = (\frac{h_2}{h_1})^{n+1}e_1 $$

## Examples

#### Example of a Taylor Series Expansion

Suppose we want to expand \\(f(x) = \cos x\\) about the point <span>\\(x_0 = 0\\)</span>. Following the formula

$$
f(x) = f(x_0)+\frac{ f'(x_0) }{1!}(x-x_0)+\frac{ f''(x_0) }{2!}(x-x_0)^2+\frac{ f'''(x_0) }{3!}(x-x_0)^3+\dotsb
$$

we need to compute the derivatives of \\(f(x) = \cos x\\) at <span>\\(x = x_0\\)</span>.

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

#### Example of Using a Truncated Taylor Series to Approximate a Function

Suppose we want to approximate \\(f(x) = \sin x\\) at <span>\\(x = 2\\)</span> using a degree-4 Taylor polynomial about (centered at) the point <span>\\(x_0 = 0\\)</span>. Following the formula

<div>\[ f(x) \approx f(x_0)+\frac{f'(x_0)}{1!}(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\frac{f'''(x_0)}{3!}(x-x_0)^3+\frac{f^{(4)}(x_0)}{4!}(x-x_0)^4, \]</div>
we need to compute the first <span>\\(4\\)</span> derivatives of \\(f(x) = \sin x\\) at <span>\\(x = x_0\\)</span>.
<div>\[\begin{align} f(x_0) &= \sin(0) = 0\\ f'(x_0) &= \cos(0) = 1\\ f''(x_0) &= -\sin(0) = 0\\ f'''(x_0) &= -\cos(0) = -1\\ f^{(4)}(x_0) &= \sin(0) = 0 \end{align}\]</div>
Then
<div>\[\begin{align} \sin x &\approx f(0)+\frac{f'(0)}{1!}x+\frac{f''(0)}{2!}x^2+\frac{f'''(0)}{3!}x^3+\frac{f^{(4)}(0)}{4!}x^4\\ &= 0 + x + 0 - \frac{1}{3!}x^3 + 0 \\ &= x - \frac{1}{6}x^3 \end{align}\]</div>
Using this truncated Taylor series centered at <span>\\(x_0 = 0\\)</span>, we can approximate \\(f(x) = \sin(x)\\) at <span>\\(x=2\\)</span>. To do so, we simply plug <span>\\(x = 2\\)</span> into the above formula for the degree 4 Taylor polynomial giving
<div>\[\begin{align} \sin(2) &\approx 2 - \frac{1}{6} 2^3 \\ &\approx 2 - \frac{8}{6} \\ &\approx \frac{2}{3} \end{align}.\]</div>

We can always use Taylor polynomial with higher degrees to do the estimation. In a similar way that we derive the closed-form expression for \\(\cos x\\), we can write the Taylor Series for \\(\sin x\\) as
<div>\[ \sin x = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)!}x^{2k+1} \]</div>

#### Example of an Error Bound

Suppose we want to approximate \\(f(x) = \sin x\\) using a degree-4 Taylor polynomial expanded about the point <span>\\(x_0 = 0\\)</span>. We want to compute the error bound for this approximation. Following Taylor Remainder Theorem,

<div>\[ R_4(x) = \frac{f^{(5)}(\xi)}{5!} (x-x_0)^{5} \]</div>
for some \\(\xi\\) between <span>\\(x_0\\)</span> and <span>\\(x\\)</span>.

If we want to find the upper bound for the absolute error, we are looking for an upper bound for $$\vert f^{(5)}(\xi)\vert $$.

Since \\(f^{(5)}(x) = \cos x\\), we have \\(|f^{(5)}(\xi)|\le 1\\). Then
\\[
|R_4(x)| = \left|\frac{f^{(5)}(\xi)}{5!} (x-x_0)^{5}\right| = \frac{|f^{(5)}(\xi)|}{5!} |x|^{5} \le \frac{1}{120} |x|^{5}
\\]

#### Example of error predictions

Suppose you expand $$\sqrt{x - 10}$$ in a Taylor polynomial of degree 3 about the center $$x_0 = 12$$. For $$h_1 = 0.5$$, you find that the Taylor truncation error is about $$10^{-4}$$.

We can use $$e_2 = (\frac{h_2}{h_1})^{n+1}e_1$$ to find the Taylor truncation error for $$h_2 = 0.25$$.

Here, $$n = 3$$ and hence $$e_2 = (\frac{0.25}{0.5})^{4} \cdot 10^{-4} = 0.625 \cdot 10^{-5}$$.

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-6-taylor.html)

## ChangeLog

- 2022-01-25 Arnav Shah [arnavss2@illinois.edu](mailto:arnavss2@illinois.edu): Added error predictions from slides
- 2021-01-20 Mariana Silva [mfsilva@illinois.edu](mailto:mfsilva@illinois.edu): Removed FD content
- 2020-02-10 Peter Sentz [sentz2@illinois.edu](mailto:sentz2@illinois.edu): Correct some small mistakes and update some notation
- 2019-01-29 John Doherty [jjdoher2@illinois.edu](mailto:jjdoher2@illinois.edu): Added Finite Difference section from F18 activity
- 2018-01-14 Erin Carrier [ecarrie2@illinois.edu](mailto:ecarrie2@illinois.edu): removes demo links
- 2017-11-02 Erin Carrier [ecarrie2@illinois.edu](mailto:ecarrie2@illinois.edu): adds changelog
- 2017-10-27 Erin Carrier [ecarrie2@illinois.edu](mailto:ecarrie2@illinois.edu): adds review questions, minor fixes throughout
- 2017-10-27 Yu Meng [yumeng5@illinois.edu](mailto:yumeng5@illinois.edu): first complete draft
- 2017-10-17 Luke Olson [lukeo@illinois.edu](mailto:lukeo@illinois.edu): outline
