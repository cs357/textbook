---
title: Rounding
description: Floating point operations have finite precision
sort: 5
author:
  - CS 357 Course Staff
changelog:
  
  - 
    name: Kaiyao Ke
    netid: kaiyaok2
    date: 2024-03-31
    message: aligned notes with slides, added examples and refactored existing notes

  - 
    name: Yuxuan Chen
    netid: yuxuan19
    date: 2022-01-30
    message: added new FP content; added FP subtraction example

  - 
    name: Mariana Silva
    netid: mfsilva
    date: 2020-04-28
    message: started from content out of FP page; added new rounding text
---
# Rounding

* * *

## Learning Objectives

*   Understand the rounding options of a non-representable number using the IEEE-754 floating pont standard.
*   Measure the error in rounding numbers using the IEEE-754 floating point standard
*   Understand catastrophic cancellation in floating point arithmetic
*   Predict the outcome of loss of significance in floating point arithmetic
*   Make an optimal selection among a group of mathematically equivalent formulae to minimize floating point error

## Rounding Options in IEEE-754

Not all real numbers can be stored exactly as floating point numbers. Consider a real number in the normalized floating point form:

$$ x = \pm 1.b_1 b_2 b_3 ... b_n ... \times 2^m $$

where $$n$$ is the number of bits in the significand and $$m$$ is the exponent for a given floating point system. If $$x$$ does not have an exact representation as a floating point number, it will be instead represented as either $$x_{-}$$ or $$x_{+}$$, the nearest two floating point numbers.

Without loss of generality, let us assume $$x$$ is a positive number. In this case, we have:

$$ x_{-} = 1.b_1 b_2 b_3 ... b_n \times 2^m $$

and

$$ x_{+} = 1.b_1 b_2 b_3 ... b_n \times 2^m  + 0.\underbrace{000000...0001}_{n\text{ bits}} \times 2^m$$

<br/>

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/rounding_basic_demo.png" width=500/> </div>

<br/>

Notice that $$x_{-}$$ is retrieved by "chopping" all the bits after the $$n^{th}$$ bit, and $$x_{+} = x_{-} + {\epsilon}_{m} \times 2^m$$, where $${\epsilon}_{m}$$ is the machine epsilon.


The process of replacing a real number $$x$$ by a nearby
machine number (either $$x_{-}$$ or $$x_{+}$$) is called **rounding**, and the error involved is called **roundoff error**.

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/rounding_line.png" width=500/> </div>
<br />
IEEE-754 doesn't specify exactly how to round floating point numbers, but there are several different options:

*   round towards zero
*   round towards infinity
*   round up
*   round down
*   round to the next nearest floating point number (either round up or down, whatever is closer)
*   round by chopping

<br />
We will denote the floating point number as $$fl(x)$$. The rounding rules above can be summarized below:

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/rounding_table.png" width=450/> </div>
&nbsp; round by chopping: $$fl(x) = x_{-}$$



## Round-off Errors

Note that the gap between two machine numbers is:

$$\vert x_{+} - x_{-} \vert = 0.\underbrace{000000...0001}_{n\text{ bits}} \times 2^m = \epsilon_m \times 2^m$$

Notice that the interval between successive floating point numbers is not uniform: the interval is smaller as the magnitude of the numbers themselves is smaller, and it is bigger as the numbers get bigger. For example, considering simple precision:

$$x_{+} \hspace{3mm}\text{and}\hspace{3mm} x_{-} \hspace{3mm}\text{of the form}\hspace{3mm} q \times 2^{-10}: \vert x_{+} - x_{-} \vert = 2^{-33} \approx 10^{-10}$$

$$x_{+} \hspace{3mm}\text{and}\hspace{3mm} x_{-} \hspace{3mm}\text{of the form}\hspace{3mm} q \times 2^{4}: \vert x_{+} - x_{-} \vert = 2^{-19} \approx 2 \times 10^{-6}$$

$$x_{+} \hspace{3mm}\text{and}\hspace{3mm} x_{-} \hspace{3mm}\text{of the form}\hspace{3mm} q \times 2^{20}: \vert x_{+} - x_{-} \vert = 2^{-3} \approx 0.125$$

$$x_{+} \hspace{3mm}\text{and}\hspace{3mm} x_{-} \hspace{3mm}\text{of the form}\hspace{3mm} q \times 2^{60}: \vert x_{+} - x_{-} \vert = 2^{37} \approx 10^{11}$$


Hence it's better to use machine epsilon to bound the error in representing a real number as a machine number.

#### Absolute error:

$$ \vert fl(x) - x \vert \le \vert x_{+} - x_{-} \vert = \epsilon_m \times 2^m$$

$$ \vert fl(x) - x \vert \le  \epsilon_m \times 2^m$$


#### Relative error:

$$ \frac{ \vert fl(x) - x \vert }{ \vert x \vert } \le \frac{ \epsilon_m \times 2^m } {  \vert \pm 1.b_1 b_2 b_3 ... b_n ... \times 2^m \vert }$$

$$ \frac{ \vert fl(x) - x \vert }{ \vert x \vert } \le \epsilon_m $$

Using this inequality, observe that for the IEEE **single-precision** floating point system:
$$ \frac{ \vert fl(x) - x \vert }{ \vert x \vert } \le 2^{-23} \approx 1.2 \times 10^{-7}$$.
Since the system consistently introduces relative errors of about $$10^{-7}$$, a single precision floating point system typically gives you about **7** (decimal) accurate digits. 

Similarly, observe that for the IEEE **double-precision** floating point system:
$$ \frac{ \vert fl(x) - x \vert }{ \vert x \vert } \le 2^{-52} \approx 2.2 \times 10^{-16}$$.
Since the system consistently introduces relative errors of about $$10^{-16}$$, a double precision floating point system typically gives you about **16** (decimal) accurate digits. 

### Example:  Determine the rounding error of a non-representable number
If we round to the nearest, what is the double-precision machine representation for 0.1? How much rounding error does it introduce?

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

Use the same algorithm that converts a decimal fraction to binary (as introduced in the previous section), we build the following table:

<table class="table">
  <thead>
    <tr>
      <th scope="col">Previous Fractional Part \( \times 2 \)</th>
      <th scope="col">Integer Part</th>
      <th scope="col">Current Fractional Part</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.2</td>
      <td>0</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>0.4</td>
      <td>0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>0.8</td>
      <td>0</td>
      <td>0.8</td>
    </tr>
    <tr>
      <td>1.6</td>
      <td>1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <td>1.2</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>0.4</td>
      <td>0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>0.8</td>
      <td>0</td>
      <td>0.8</td>
    </tr>
    <tr>
      <td>1.6</td>
      <td>1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <td>1.2</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table> 

We can observe that the integer part sequence  \( {0, 0, 1, 1} \) is recurring.<br>

Then, we have \( 0.1 = (0.000110011\hspace{1mm}\overline{0011})_2 = (1.10011\hspace{1mm}\overline{0011})_2 \times 2^{-4}\) .<br>

Rounded by nearest, the double-precision machine representation for 0.1 is the tuple \( (s, c, f) \), where \(s\) is the sign bit, \(c\) is the exponent field, and \(f\) is the significand: <br>
\( s = 0 \)<br>
\( m = -4 \Rightarrow  c = m + 1023 = 1019 = 01111111011\)<br>
\( f = 10011...0011...0011010 \)<br>

In other words, the double-precision machine representation for 0.1 is:<br>

\( \bf{0\hspace{2mm}01111111011\hspace{2mm}\underbrace{10011...0011...0011010}_{52\text{ bits}}} \)<br>

The rounding error is then:<br>
\( fl(0.1) - 0.1 = (0.00011001100110011...11010)_2 - (0.00011001100110011...1100110011\overline{0011})_2\)
\(= (0.\underbrace{0...0}_{54\text{ 0's}}000110011\overline{0011})_2 = 2^{-54} \times 0.1 = 2^{-1} \times 2^{-53} \times 0.1 = \bf{0.05\epsilon}\)<br>
</details>


### Example: Floating Point Equality
Assume you are working with IEEE single-precision numbers. Find the smallest representable number $$a$$ that satisfies $$2^8 + a \neq 2^8$$.
<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

The smallest representable number larger than \(2^8\) is \((2+\epsilon_m)\times 2^8\). <br>
Let \(x_{-} = 2^8\) and \(x_{+} = (2+\epsilon_m)\times 2^8\). <br>
In order to have \(2^8 + a\) rounded to \(x_{+}\), we must have \(a \gt \frac{\vert x_{+} - x_{-} \vert}{2} = \frac{\epsilon_m\times 2^8}{2} = \frac{2^{-23}\times 2^8}{2} = 2^{-16}\)<br>
The smallest machine representable number greater than \(2^{-16}\) is \((1 + \epsilon_m) \times 2^{-16}\)<br>
Hence \(\bf{a = \bf{(1 + 2^{-23}) \times 2^{-16}}}\)
</details>


### Example: Uneven Distribution of Floating Point Numbers in any FPS
Consider the small floating point number system $$ x = \pm 1.b_1 b_2 \times 2^m \) for \(m \in [-4, 4]\) and \(b_i \in \{0, 1\}$$:

<details style="line-height: 2;">
    <summary><strong>Details</strong></summary>

\( (1.00)_2 \times 2^{-4} = 0.0625 \)<br>
\( (1.01)_2 \times 2^{-4} = 0.078125 \)<br>
\( (1.10)_2 \times 2^{-4} = 0.09375 \)<br>
\( (1.11)_2 \times 2^{-4} = 0.109375 \)<br>
\( (1.00)_2 \times 2^{-3} = 0.125 \)<br>
\( (1.01)_2 \times 2^{-3} = 0.15625 \)<br>
\( (1.10)_2 \times 2^{-3} = 0.1875 \)<br>
\( (1.11)_2 \times 2^{-3} = 0.21875 \)<br>
\( (1.00)_2 \times 2^{-2} = 0.25 \)<br>
\( (1.01)_2 \times 2^{-2} = 0.3125 \)<br>
\( (1.10)_2 \times 2^{-2} = 0.375 \)<br>
\( (1.11)_2 \times 2^{-2} = 0.4375 \)<br>
\( (1.00)_2 \times 2^{-1} = 0.5 \)<br>
\( (1.01)_2 \times 2^{-1} = 0.625 \)<br>
\( (1.10)_2 \times 2^{-1} = 0.75 \)<br>
\( (1.11)_2 \times 2^{-1} = 0.875 \)<br>
\( (1.00)_2 \times 2^{0} = 1.0 \)<br>
\( (1.01)_2 \times 2^{0} = 1.25 \)<br>
\( (1.10)_2 \times 2^{0} = 1.5 \)<br>
\( (1.11)_2 \times 2^{0} = 1.75 \)<br>
\( (1.00)_2 \times 2^{1} = 2.0 \)<br>
\( (1.01)_2 \times 2^{1} = 2.5 \)<br>
\( (1.10)_2 \times 2^{1} = 3.0 \)<br>
\( (1.11)_2 \times 2^{1} = 3.5 \)<br>
\( (1.00)_2 \times 2^{2} = 4.0 \)<br>
\( (1.01)_2 \times 2^{2} = 5.0 \)<br>
\( (1.10)_2 \times 2^{2} = 6.0 \)<br>
\( (1.11)_2 \times 2^{2} = 7.0 \)<br>
\( (1.00)_2 \times 2^{3} = 8.0 \)<br>
\( (1.01)_2 \times 2^{3} = 10.0 \)<br>
\( (1.10)_2 \times 2^{3} = 12.0 \)<br>
\( (1.11)_2 \times 2^{3} = 14.0 \)<br>
\( (1.00)_2 \times 2^{4} = 16.0 \)<br>
\( (1.01)_2 \times 2^{4} = 20.0 \)<br>
\( (1.10)_2 \times 2^{4} = 24.0 \)<br>
\( (1.11)_2 \times 2^{4} = 28.0 \)<br>
</details>




## Mathematical Properties of Floating Point Operations

*   Not necessarily associative: $$ (x + y) + z \neq x + (y + z) $$.  

This is because $$ fl(fl(x + y) + z) \neq fl(x + fl(y + z)) $$.
*   Not necessarily distributive: $$ z \cdot (x + y) \neq z \cdot x + z \cdot y $$.  

This is because $$ fl(z \cdot fl(x + y)) \neq fl(fl(z \cdot x) + fl(z \cdot y)) $$.
*   Not necessarily cumulative: repeatedly adding a very small number to a large number may do nothing.

### Example: Non-associative Feature

Find $$a$$, $$b$$ and $$c$$ such that $$(a + b) + c \neq a + (b + c)$$ under double-precision floating point arithmetic.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

Let \(a = \pi, b = 10^{100}, c = -10^{100}\)<br>
Under double-precision floating point arithmetic:<br>
\((a + b) + c = 10^{100} + (-10^{100}) = 0\)<br>
\(a + (b + c) = \pi + 0 = \pi\)

</details>


### Example: Non-distributive Feature

Find $$a$$, $$b$$ and $$c$$ such that $$c(a + b) \neq ca + cb$$ under double-precision floating point arithmetic.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

Let \(a = 0.1, b = 0.2, c = 100\)<br>
Under double-precision floating point arithmetic:<br>
\(c(a + b) = 100 \times 0.30000000000000004 = 30.000000000000004\)<br>
\(ca + cb = 10 + 20 = 30\)

</details>


## Floating Point Arithmetic

The basic idea of floating point arithmetic is to first compute the exact result, and then round the result to make it fit into the desired precision:

$$x + y = fl(x + y)$$

$$x \times y = fl(x \times y)$$


## Floating Point Addition

Adding two floating point numbers is easy. The basic idea is:

1.  Bring both numbers to a common exponent
2.  Do grade-school addition from the front, until you run out of digits in your system
3.  Round the result

Consider a number system such that $$ x = \pm 1.b_1 b_2 b_3 \times 2^m $$ for $$m \in [-4, 4]$$ and $$b_i \in \{0, 1\}$$, and we perform addition on two numbers from this system.

### Example 1: No Rounding Needed

Let $$a = (1.101)_2 \times 2^1$$ and $$b = (1.001)_2 \times 2^1$$, find $$c = a + b$$.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

\(c = a + b = (10.110)_2 \times 2^1 = (1.011)_2 \times 2^2\)<br>

</details>


### Example 2: No Rounding Needed

Let $$a = (1.100)_2 \times 2^1$$ and $$b = (1.100)_2 \times 2^{-1}$$, find $$c = a + b$$.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

\(c = a + b = (1.100)_2 \times 2^1 + (0.011)_2 \times 2^1 = (1.111)_2 \times 2^1\)<br>

</details>


### Example 3: Requiring Rounding and Precision Lost

Let $$a = (1.a_1a_2a_3a_4a_5a_6...a_n...)_2 \times 2^0$$ and $$b = (1.b_1b_2b_3b_4b_5b_6...b_n...)_2 \times 2^{-8}$$, find $$c = a + b$$. Assume a single-precision system is used, and that $$n \gt 23$$.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

In single precision: <br>
\(a = (1.a_1a_2a_3a_4a_5a_6...a_{22}a_{23})_2 \times 2^0\)<br>
\(b = (1.b_1b_2b_3b_4b_5b_6...b_{22}b_{23})_2 \times 2^{-8}\)<br>
So \(a + b = (1.a_1a_2a_3a_4a_5a_6...a_{22}a_{23})_2 \times 2^0 + (0.00000001b_1b_2b_3b_4b_5b_6...b_{14}b_{15})_2 \times 2^0\)<br>

In this example, the result \(c = fl(a+b)\) includes only \(15\) bits of precision from \(b\), so a lost of precision happens as well.

</details>


### Example 4: Require Rounding and Precision Lost

Let $$a = (1.101)_2 \times 2^0$$ and $$b = (1.000)_2 \times 2^0$$, find $$c = a + b$$. Round down if necessary.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

\(c = a + b = (10.101)_2 \times 2^0 \approx (1.010)_2 \times 2^1\)<br>

</details>


### Example 5: Require Rounding and Precision Lost

Let $$a = (1.101)_2 \times 2^1$$ and $$b = (1.001)_2 \times 2^{-1}$$, find $$c = a + b$$. Round down if necessary.

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

<div>\[\begin{align} a &= 1.101 \times 2^1 \\ b &= 0.01001 \times 2^1 \\ a + b &= 1.111 \times 2^1 \\ \end{align}\]</div>

You'll notice that we added two numbers with 4 significant digits, and our result also has 4 significant digits. There is no loss of significant digits with floating point addition.

</details>


### Example 6: No Rounding Needed but Precision Lost

Let $$a = (1.1011)_2 \times 2^1$$ and $$b = (-1.1010)_2 \times 2^1$$, find $$c = a + b$$. 

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

\(c = a + b = (0.0001)_2 \times 2^1 = (1.????)_2 \times 2^{-3}\)<br>

Unfortunately there is not data to indicate what the missing digits should be. The effect is that the number of significant digits in the result is reduced. Machine fills them with its best guess,which is often not good (usually what is called spurious zeros). This phenomenon is called Catastrophic Cancellation.

</details>

## Floating Point Subtraction and Catastrophic Cancellation

Floating point subtraction works much the same was that addition does. However, problems occur when you subtract two numbers of similar magnitude.

For example, in order to subtract \\(b = (1.1010)_2 \times 2^1\\) from \\(a = (1.1011)_2 \times 2^1\\), this would look like:

<div>\[\begin{align} a &= 1.1011???? \times 2^1 \\ b &= 1.1010???? \times 2^1 \\ a - b &= 0.0001???? \times 2^1 \\ \end{align}\]</div>

When we normalize the result, we get \\(1.???? \times 2^{-3}\\). There is no data to indicate what the missing digits should be. Although the floating point number will be stored with 4 digits in the fractional, it will only be accurate to a single significant digit. This loss of significant digits is known as **_catastrophic cancellation_**.

More rigidly, consider a general case when $$x \approx y$$. Without loss of generality, we assume $$x, y \gt 0$$ (if $$x$$ and $$y$$ are negative, then $$y - x = -(x - y) = -x - (-y)$$, where $$-x$$ and $$-y$$ are both positive and similar in magnitude). Suppose we need to calculate $$fl(y - x)$$ given:

$$ x = 1.a_1 a_2 a_3 ... a_{n-2} a_{n-1}0 a_{n+1} a_{n+2} ... \times 2^m $$

$$ y = 1.a_1 a_2 a_3 ... a_{n-2} a_{n-1}1 b_{n+1} b_{n+2} ... \times 2^m $$

It can be shown that for all $$n \gt 1$$, precision lost happens, and it becomes more catastrophic when $$n$$ increases (in other words, when $$x$$ and $$y$$ gets more similar). Then, consider when $$n = 23$$ and single-precision floating point system is used, then $$fl(y - x) = 1.????... \times 2^{-23+m}$$, where the leading "$$1$$" before the decimal point becomes the only significant digit. Notice that the floating point system may produce $$fl(y - x) = 1.000... \times 2^{-n+m}$$, but all digits after the decimal point are "guessed" and are not significant digits. In fact, the precision lost is not due to $$fl(y-x)$$ but due to rounding of $$x, y$$ from the beginning in order to get numbers representable by the floating point system.

#### Example: Using Mathematically-Equivalent Formulae to Prevent Cancellation
Consider the function $$ f(x) = \sqrt{x^{2} + 1} - 1 $$. When we evaluate $$ f(x) $$ for values of $$ x $$ near zero, we may encounter loss of significance due to floating point subtraction. If $$ x = 10^{-3} $$, using five-decimal-digit arithmetic, $$ f(10^{-3}) = \sqrt{10^{-6} + 1} - 1 = 0 $$. How can we find an alternative formula to perform mathematically-equivalent computation without catastrophic cancellation?

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

A method of avoiding loss of significant digits is to eliminate subtraction: <br>

\( f(x) = \sqrt{x^{2} + 1} - 1 = \frac{ (\sqrt{x^{2} + 1} - 1) \cdot (\sqrt{x^{2} + 1} + 1) } { \sqrt{x^{2} + 1} + 1 }  = \frac{ x^{2} } { (\sqrt{x^{2} + 1} + 1) } \) <br>

Thus for \( x = 10^{-3} \), using five-decimal-digit arithmetic, \( f(10^{-3}) = \frac{ 10^{-6} } { 2 } \).

</details>


#### Example: Calculating Relative Error when Cancellation Occurs
If $$x = 0.3721448693$$ and $$y = 0.3720214371$$, what is the relative error in the computation of
$$(x − y)$$ in a computer with five decimal digits of accuracy?

<details style="line-height: 2;">
    <summary><strong>Answer</strong></summary>

Using five decimal digits of accuracy, the numbers are rounded as: <br>

\(fl(x) = 0.37214\) and \(fl(y) = 0.37202\) <br>

Then the subtraction is computed: <br>

\(fl(x) − fl(y) = 0.37214 − 0.37202 = 0.00012 \) <br>

The result of the operation is: <br>

\(fl(x − y) = 1.20000 × 10^{−2}\) <br>

Notice that the last four digits are filled with spurious zeros. <br>

The relative error between the exact and computer solutions is given by: <br>

\( \frac{\vert (x - y) - fl(x - y) \vert}{\vert (x - y) \vert} = \frac{0.0001234322 − 0.00012}{0.000123432} = \frac{0.0000034322}{0.000123432} \approx \bf{3 \times 10^{-2}} \) <br>

Note that the magnitude of the error due to the subtraction is large when compared with the relative
error due to the rounding, which is: <br>

\( \frac{\vert x - fl(x) \vert}{\vert x \vert} \approx 1.3 \times 10^{-5} \)

</details>

## Review Questions
<ol>
  <li> Given a real number, what is the rounding error involved in storing it as a machine number? What is the relative error?</li>
  <li> How can we bound the relative error of representing a real number as a normalized machine number?</li>
  <li> What is cancellation? Why is it a problem?</li>
  <li> What types of operations lead to catostrophic cancellation?</li>
  <li> Given two different equations for evaluating the same value, can you identify which is more accurate for certain x and why?</li>
</ol>
