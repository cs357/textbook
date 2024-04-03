---
title: Floating Point Representation
description: A way to store approximations of real numbers in silicons.
sort: 4
author:
  - CS 357 Course Staff
changelog:
  - 
    name: Apramey Hosahalli
    netid: apramey2
    date: 2024-04-02
    message: adjust examples, reorder sections to match slides
  - 
    name: Apramey Hosahalli
    netid: apramey2
    date: 2024-04-02
    message: add integer range section and fixed machine epsilon
  - 
    name: Apramey Hosahalli
    netid: apramey2
    date: 2024-03-31
    message: add binary point location section
  - 
    name: Apramey Hosahalli
    netid: apramey2
    date: 2024-03-27
    message: update learning objectives and add section for fixed-point representation
  - 
    name: Mariana Silva
    netid: mfsilva
    date: 2020-04-28
    message: moved rounding content to a separate page
  - 
    name: Wanjun Jiang
    netid: wjiang24
    date: 2020-01-26
    message: add normalized fp numbers, and some examples

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-01-14
    message: removes demo links

  - 
    name: Adam Stewart
    netid: adamjs5
    date: 2017-12-13
    message: fix incorrect formula under number systems and bases

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-12-08
    message: specifies UFL is positive

  - 
    name: Matthew West
    netid: mwest
    date: 2017-11-19
    message: addition of machine epsilon diagrams

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-18
    message: updates machine eps def

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-15
    message: fixes typo in converting integers

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-14
    message: clarifies when stored normalized
  
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-13
    message: updates machine epsilon definition, fixes inconsistent capitalization
  
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-13
    message: updates machine epsilon

  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2017-11-12
    message: minor fixes throughout, adds changelog, adds section on number systems in different bases

  - 
    name: Adam Stewart
    netid: adamjs5
    date: 2017-11-01
    message: first complete draft

  - 
    name: Luke Olson
    netid: lukeo
    date: 2017-10-16
    message: outline

---

# Floating Point Representation

* * *

## Learning Objectives

*   Represent numbers in floating point systems
*   Evaluate the range, precision, and accuracy of different representations
*   Define Machine Epsilon
*   Identify the smallest and largest representable floating point number
*   Handle special cases and subnormal numbers


## Number Systems and Bases

There are a variety of number systems in which a number can be represented. In the common base 10 (decimal) system each digit takes on one of 10 values, from 0 to 9\. In base 2 (binary), each digit takes on one of 2 values, 0 or 1.

For a given \\(\beta\\), in the \\(\beta\\)-system we have:
<div>\[(a_n \ldots a_2 a_1 a_0 . b_1 b_2 b_3 b_4 \dots)_{\beta} = \sum_{k=0}^{n} a_k \beta^k + \sum_{k=1}^\infty b_k \beta^{-k}.\]</div>


Some common bases used for numbering systems are:

*   decimal: \\(\beta=10\\)
*   binary: \\(\beta=2\\)
*   octal: \\(\beta=8\\)
*   hexadecimal \\(\beta=16\\)

### Example:

Decimal base: $$(426.97)_{10} $$

<details markdown="1">

<summary><strong>Answer</strong></summary>
$$\begin{equation}(426.97)_{10} = 4 \times 10^2 + 2 \times 10^1 + 6 \times 10^0 + 9 \times 10^{-1} + 7 \times 10^{-2} \end{equation} $$
</details>

Binary base:$$(1011.001)_{2}$$

<details markdown="1">

<summary><strong>Answer</strong></summary>
$$(1011.001)_{2} = 1 \times 2^3 + 0 \times 2^2 + 1 \times 2^1 + 1 \times 2^0 + 0 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} = (11.125)_{10}$$
</details>


## Converting Integers Between Decimal and Binary

Modern computers use transistors to store data. These transistors can either be ON (1) or OFF (0). In order to store integers in a computer, we must first convert them to binary. For example, the binary representation of 23 is <span>\\((10111)_2\\)</span>.

Converting an integer from binary representation (base 2) to decimal representation (base 10) is easy. Simply multiply each digit by increasing powers of 2 like so:

<div>\[(10111)_2 = 1 \cdot 2^4 + 0 \cdot 2^3 + 1 \cdot 2^2 + 1 \cdot 2^1 + 1 \cdot 2^0 = 23\]</div>

Converting an integer from decimal to binary is a similar process, except instead of multiplying by 2 we will be dividing by 2 and keeping track of the remainder:

<div>\[\begin{align} 23 // 2 &= 11\ \mathrm{rem}\ 1 \\ 11 // 2 &= 5\ \mathrm{rem}\ 1 \\ 5 // 2 &= 2\ \mathrm{rem}\ 1 \\ 2 // 2 &= 1\ \mathrm{rem}\ 0 \\ 1 // 2 &= 0\ \mathrm{rem}\ 1 \\ \end{align}\]</div>

Thus, $$(23)_{10}$$ becomes <span>\\((10111)_2\\)</span> in binary.


You may find these additional resources helpful for review: [Decimal to Binary 1](https://www.wikihow.com/Convert-from-Decimal-to-Binary) and [Decimal to Binary 2](http://interactivepython.org/courselib/static/pythonds/BasicDS/ConvertingDecimalNumberstoBinaryNumbers.html)


## Converting Fractions Between Decimal and Binary

Real numbers add an extra level of complexity. Not only do they have a leading integer, they also have a fractional part. For now, we will represent a decimal number like 23.375 as <span>\\((10111.011)_2\\)</span>. Of course, the actual machine representation depends on whether we are using a fixed point or a floating point representation, but we will get to that in later sections.

Converting a number with a fractional portion from binary to decimal is similar to converting to an integer, except that we continue into negative powers of 2 for the fractional part:

<div>\[(10111.011)_2 = 1 \cdot 2^4 + 0 \cdot 2^3 + 1 \cdot 2^2 + 1 \cdot 2^1 + 1 \cdot 2^0 + 0 \cdot 2^{-1} + 1 \cdot 2^{-2} + 1 \cdot 2^{-3} = 23.375\]</div>

To convert a decimal fraction to binary, first convert the integer part to binary as previously discussed. Then, take the fractional part (ignoring the integer part) and multiply it by 2\. The resulting integer part will be the binary digit. Throw away the integer part and continue the process of multiplying by 2 until the fractional part becomes 0\. For example:

<div>\[\begin{align} 23 &= (10111)_2 \\ 2 \cdot .375 &= 0.75 \\ 2 \cdot .75 &= 1.5 \\ 2 \cdot .5 &= 1.0 \\ \end{align}\]</div>

By combining the integer and fractional parts, we find that <span>\\(23.375 = (10111.011)_2\\)</span>.

Not all fractions can be represented in binary using a finite number of digits. For example, if you try the above technique on a number like 0.1, you will find that the remaining fraction begins to repeat:

<div>\[\begin{align} 2 \cdot .1 &= 0.2 \\ 2 \cdot .2 &= 0.4 \\ 2 \cdot .4 &= 0.8 \\ 2 \cdot .8 &= 1.6 \\ 2 \cdot .6 &= 1.2 \\ 2 \cdot .2 &= 0.4 \\ 2 \cdot .4 &= 0.8 \\ 2 \cdot .8 &= 1.6 \\ 2 \cdot .6 &= 1.2 \\ \end{align}\]</div>

As you can see, the decimal 0.1 will be represented in binary as the infinitely repeating series <span>\\((0.00011001100110011...)_2\\)</span>. The exact number of digits that get stored in a floating point number depends on whether we are using single precision or double precision.

<!-- Another resource for review: [Decimal Fraction to Binary](http://cs.furman.edu/digitaldomain/more/ch6/dec_frac_to_bin.htm) -->

## (Unsigned) Fixed-Point Representation

In fixed-point representation, numbers are stored with a fixed number of bits for the integer part and a fixed number of bits for the fractional part.

Suppose we have 64 bits to store a real number, where 32 bits store the integer part and 32 bits store the fractional part.

$$ (a_{31}...a_2a_1a_0.b_1b_2b_3...b_{32}) = \sum_{k=0}^{31} a_k 2^k + \sum_{k=1}^{32} b_k 2^{-k} $$

$$\begin{eqnarray}
 = &a_{31}& \times 2^{31}\ + ... + a_1 \times 2^1 + a_0 \times 2^0 \\
+ &b_1& \times 2^{-1} + b_2 \times 2^{-2}+ ... +\ b_{32} \times 2^{-32}
\end{eqnarray}$$

**Smallest Number** \\
We get the smallest representable number when all of the values of $$a_i$$ and $$b_i$$ are $$0$$, except for $$b_{32}$$.

$$\begin{eqnarray}
&a_i =& 0 \ \forall i,\ b_1,\ b_2, ...,\ b_{31} = 0,\ and \ b_{32} = 1 \\
&\implies& 2^{-32} \approx 10^{-9}
\end{eqnarray}$$

**Largest Number** \\
We get the largest representable number when all of the values of $$a_i$$ and $$b_i$$ are $$1$$.

$$\begin{eqnarray}
&a_i =& 1 \ \forall i \ and \ b_i = 1 \ \forall i \\
&\implies&  2^{31}\ + ... + 2^1 + 2^0 + 2^{-1} + 2^{-2}+ ... +\ 2^{-32} \approx 10^9
\end{eqnarray}$$

### Binary Point Placement

We consider the range and precision when evaluating a representation.

**Range:** Difference between the largest and smallest numbers possible.

**Precision:** Smallest possible difference between any two numbers.

Consider an 8 bit system where 6 bits store the integer part and 2 bits store the fractional part. What are its range and precision?

<details markdown="1">

<summary><strong>Answer</strong></summary>

**Smallest Numbers** \\
$$000000.01 = 0.25$$ \\
$$000000.10 = 0.5$$

**Largest Numbers** \\
$$111111.10 = 63.5$$ \\
$$111111.11 = 63.75 $$

**Range** \\
$$63.75 - 0.25 = 63.5$$

**Precision** \\
$$0.5 - 0.25 = 63.75 - 63.5 = 0.25$$

</details>


How about for an 8 bit system where 2 bits store the integer part and 6 bits store the fractional part?

<details markdown="1">

<summary><strong>Answer</strong></summary>

**Smallest Numbers** \\
$$00.000001 = 0.015625$$ \\
$$00.000010 = 0.03125$$

**Largest Numbers** \\
$$11.111110 = 3.96875$$ \\
$$11.111111 = 3.984375 $$

**Range** \\
$$3.984375 - 0.015625 = 3.96875$$

**Precision** \\
$$0.03125 - 0.015625 = 3.984375 - 3.96875 = 0.015625$$

</details>

We see that having more **integer** bits increases **range** and more bits for the **fractional** part increases **precision**.
It can be hard to decide how much precision and range you need.

**Fix: Let the binary point "float"**

## Floating Point Numbers

The floating point representation of a binary number is similar to scientific notation for decimals. Much like you can represent 23.375 as,

<div>\[2.3375 \cdot 10^1\]</div>

you can represent <span>\\((10111.011)_2\\)</span> as,

<div>\[1.0111011 \cdot 2^4\]</div>.

A floating-point number can represent numbers of different orders of magnitude(very large and very small) with the same number of fixed digits.

More formally, we can define a floating point number <span>\\(x\\)</span> as,

<div>\[x = \pm q \cdot 2^m\]</div>

where,

*   \\(\pm\\) is the sign
*   <span>\\(q\\)</span> is the significand
*   <span>\\(m\\)</span> is the exponent.

<br/>
Aside from the special case of zero and subnormal numbers (discussed below), the significand is always in normalized form,

<div>\[q = 1.f\]</div>

where,

*   <span>\\(f\\)</span> is the fractional part of the significand.

<br/>
Whenever we store a normalized floating point number, the 1 is assumed. We don't store the entire significand, just the fractional part. This is called the "hidden bit representation", which gives one additional bit of precision.

## Properties of Normalized Floating-Point Systems

A number <span>\\(x\\)</span> in a normalized binary floating-point system has the form
<div>\[ \begin{equation} x = \pm 1.b_1b_2b_3...b_n \times 2^m = \pm 1.f \times 2^m \end{equation} \]</div>

*   **Digits:** \\(b_i \in \{0, 1\}\\)
*   **Exponent range:** Integer \\(m \in [L,U]\\)
*   **Precision:** <span>\\(p = n + 1\\)</span>
*   **Smallest positive normalized floating-point number:** <span>\\( 2^L\\)</span>
*   **Largest positive normalized floating-point number:** <span>\\( 2^{U+1}(1-2^{-p})\\)</span>

<br/>

### Example

<div>\[\begin{equation} x = \pm 1.b_1b_2 \times 2^m \text{ for } m \in [-4,4] \text{ and } b_i \in \{0,1\} \end{equation} \]</div>

<details markdown="1">

<summary><strong>Answer</strong></summary>
*   Smallest normalized positive number:
<div>\[ \begin{equation} (1.00)_2 \times 2^{-4} = 0.0625 \end{equation} \]</div>
*   Largest normalized positive number:
<div>\[ \begin{equation} (1.11)_2 \times 2^4 = 28.0 \end{equation} \]</div>
*   Any number <span>\\(x\\)</span> closer to zero than 0.0625 would underflow to zero.
*   Any number <span>\\(x\\)</span> outside the range -28.0 and +28.0 would overflow to infinity.

</details>

### Machine Epsilon

**_Machine epsilon_** (\\(\epsilon_m\\)) is defined as the distance (gap) between $$1$$ and the next largest floating point number.

$$\pm 1.b_1b_2 \times 2^m\ for \ m \subset [-4,4]\ and\ b_i \subset {0, 1}$$

$$(1.00)_2 \times 2^0 = 1 \hspace{1.8cm} (1.01)_2 \times 2^0 = 1.25 $$

$$ \epsilon_m = (0.01)_2 \times 2^0 = 0.25 $$

Or for a general normalized floating point system $$1.f \times 2^m$$, where $$f$$ is represented with $$n$$ bits, machine epsilon is defined as:

$$ \epsilon_m = 2^{-n} $$

In programming languages these values are typically available as predefined constants.
For example, in C, these constants are `FLT_EPSILON` and `DBL_EPSILON` and are defined in the `float.h` library.
In Python you can access these values with the code snippet below.

```python
import numpy as np
# Single Precision
eps_single = np.finfo(np.float32).eps
print("Single precision machine eps = {}".format(eps_single))
# Double Precision
eps_double = np.finfo(np.float64).eps
print("Double precision machine eps = {}".format(eps_double))
```

_Note:_ There are many definitions of machine epsilon that are used in various resources, such as the smallest number such that \\(\text{fl}(1 + \epsilon_m) \ne 1\\). These other definitions may give slightly different values from the definition above depending on the rounding mode (next topic). In this course, we will always use the values from the "gap" definition above.

### Range of Integer Numbers

What is the range of integer numbers that can be represented exactly in this representation?

$$\pm 1.b_1b_2 \times 2^m\ for \ m \subset [-4,4]\ and\ b_i \subset {0, 1}$$

$$(1)_2 = 1.00 \times 2^0 = 1_{10}$$ \\
$$(10)_2 = 1.00 \times 2^1 = 2_{10}$$ \\
$$(11)_2 = 1.10 \times 2^1 = 3_{10}$$ \\
$$(100)_2 = 1.00 \times 2^2 = 4_{10}$$ \\
$$(101)_2 = 1.00 \times 2^2 = 5_{10}$$ \\
$$(110)_2 = 1.00 \times 2^2 = 6_{10}$$ \\
$$(111)_2 = 1.00 \times 2^2 = 7_{10}$$ \\
$$(1000)_2 = 1.00 \times 2^3 = 8_{10}$$ \\
$$\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ = 9_{10}$$ \\
$$(1001)_2 = 1.00 \times 2^3 = 10_{10}$$

The limit in precision causes skips in integers above an integer range. The upper bound of this range is given by $$2^p$$.


## IEEE-754 Single Precision

![Single Precision]({{ site.baseurl }}/assets/img/figs/ieee_single.png)

*   \\(x = (-1)^s 1.f \times 2^m\\)
*   1-bit sign, s = 0: positive sign, s = 1: negative sign
*   8-bit exponent <span>\\(c\\)</span>, where <span>\\(c = m + 127\\)</span>, we need to reserve exponent number for special cases <span>\\( c = (11111111)_2 = 255, c = (00000000)_2 = 0\\)</span>, therefore <span class="math"><span>\\(0 < c < 255\\)</span></span>
*   23-bit fractional part <span>\\(f\\)</span>
*   Machine epsilon: 
    * For IEEE-754 **single precision**, \\(\epsilon_m = 2^{-23}\\), as shown by:

        $$
           \epsilon_m = 1.\underbrace{000000...000}_{\text{22 bits}}{\bf 1} - 1.\underbrace{000000...000}_{\text{22 bits}}{\bf 0} = 2^{-23}
        $$
*   Smallest positive normalized FP number: \\(UFL = 2^L = 2^{-126} \approx 1.2 \times 10^{-38}\\)
*   Largest positive normalized FP number: \\(OFL = 2^{U+1}(1 - 2^{-p}) = 2^{128}(1 - 2^{-24}) \approx 3.4 \times 10^{38}\\)


<br/>
The exponent is shifted by 127 to avoid storing a negative sign. Instead of storing <span>\\(m\\)</span>, we store <span>\\(c = m + 127\\)</span>. Thus, the largest possible exponent is 127, and the smallest possible exponent is -126.

### Example:

Convert the binary number $$(100101.101)_2$$ to the normalized FP representation

$$1.f \times 2^m$$

<details markdown="1">

<summary><strong>Answer</strong></summary>
$$(100101.101)_2 = (1.00101101)_2 \times 2^5 $$
$$ s = 0,\quad f = 00101101…00,\quad m = 5 $$
$$ c=m + 127 = 132 = (10000100)_2 $$

Normalized FP representations:  $$ 0 \; 10000100 \; 00101101000000000000000  $$ 
</details>
For additional reading about [IEEE Floating Point Numbers](http://steve.hollasch.net/cgindex/coding/ieeefloat.html)

## IEEE-754 Double Precision

![Double Precision]({{ site.baseurl }}/assets/img/figs/ieee_double.png)

*   \\(x = (-1)^s 1.f \times 2^m\\)
*   1-bit sign, s = 0: positive sign, s = 1: negative sign
*   11-bit exponent <span>\\(c\\)</span>, where <span>\\( c = m + 1023\\)</span>, we need to reserve exponent number for special cases <span>\\( c = (11111111111)_2 = 2047, c = (00000000000)_2 = 0\\)</span>, therefore <span class="math"><span>\\(0 < c < 2047\\)</span></span>
*   52-bit fractional part <span>\\(f\\)</span>
*   Machine epsilon:
    * For IEEE-754 **double precision**, \\(\epsilon_m = 2^{-52}\\), as shown by:

    $$
    \epsilon_m = 1.\underbrace{000000...000}_{\text{51 bits}}{\bf 1} - 1.\underbrace{000000...000}_{\text{51 bits}}{\bf 0} = 2^{-52}
    $$
*   Smallest positive normalized FP number: \\(UFL = 2^L = 2^{-1022} \approx 2.2 \times 10^{-308}\\)
*   Largest positive normalized FP number: \\(OFL = 2^{U+1}(1 - 2^{-p}) = 2^{1024}(1 - 2^{-53}) \approx 1.8 \times 10^{308}\\)

<br/>
The exponent is shifted by 1023 to avoid storing a negative sign. Instead of storing <span>\\(m\\)</span>, we store <span>\\(c = m + 1023\\)</span>. Thus, the largest possible exponent is 1023, and the smallest possible exponent is -1022.

## Special Cases in IEEE-754

There are several corner cases that arise in floating point representations.

### Zero

In our definition of floating point numbers above, we said that there is always a leading 1 assumed. This is true for most floating point numbers. A notable exception is zero. In order to store zero as a floating point number, we store all zeros for the exponent and all zeros for the fractional part. Note that there can be both +0 and -0 depending on the sign bit.

### Infinity

If a floating point calculation results in a number that is beyond the range of possible numbers in floating point, it is considered to be infinity. We store infinity with all ones in the exponent and all zeros in the fractional. \\(+\infty\\) and \\(-\infty\\) are distinguished by the sign bit.

### NaN

Arithmetic operations that result in something that is not a number are represented in floating point with all ones in the exponent and a non-zero fractional part.

## Floating Point Number Line

![Number Line]({{ site.baseurl }}/assets/img/figs/floatingpoints.png)

The above image shows the number line for the IEEE-754 floating point system.

## Subnormal Numbers

A **_normal number_** is defined as a floating point number with a 1 at the start of the significand. Thus, the smallest normal number in double precision is \\(1.000... \times 2^{-1022}\\). The smallest representable _normal_ number is called the **_underflow level_**, or **_UFL_**.

However, we can go even smaller than this by removing the restriction that the first number of the significand must be a 1\. These numbers are known as **_subnormal_**, and are stored with all zeros in the exponent. Technically, zero is also a subnormal number.

It is important to note that subnormal numbers do not have as many significant digits as normal numbers.

**IEEE-754 Single precision (32 bits):**

*   <span>\\( c = (00000000)_2 = 0 \\)</span>
*   Exponent set to <span>\\(m\\)</span> = -126
*   Smallest positive subnormal FP number: \\(2^{-23} \times 2^{-126} \approx 1.4 \times 10^{-45}\\)

<br/>
**IEEE-754 Double precision (64 bits):**

*   <span>\\( c = (00000000000)_2 = 0 \\)</span>
*   Exponent set to <span>\\(m\\)</span> = -1022
*   Smallest positive subnormal FP number: \\(2^{-52} \times 2^{-1022} \approx 4.9 \times 10^{-324}\\)

<br/>
The use of subnormal numbers allows for more gradual underflow to zero (however subnormal numbers don't have as many accurate bits as normalized numbers).



## Review Questions

<ol class="review">
<li> Convert a decimal number to binary.</li>
<li> Convert a binary number to decimal.</li>
<li> What are the differences between floating point and fixed point representation?</li>
<li> Given a real number, how would you store it as a machine number?</li>
<li> Given a real number, what is the rounding error involved in storing it as a machine number? What is the relative error?</li>
<li> Explain the different parts of a floating-point number: sign, significand, and exponent.</li>
<li> How is the exponent of a machine number actually stored?</li>
<li> What is machine epsilon?</li>
<li> What is underflow (UFL)?</li>
<li> What is overflow?</li>
<li> Why is underflow sometimes not a problem?</li>
<li> Given a toy floating-point system, determine machine epsilon and UFL for that system.</li>
<li> How do you store zero as a machine number?</li>
<li> What are subnormal numbers?</li>
<li> How are subnormal numbers represented in a machine?</li>
<li> Why are subnormal numbers sometimes helpful?</li>
<li> What are some drawbacks to using subnormal numbers?</li>
</ol>