---
title: Random Number Generators and Monte Carlo Method
description: Properties of RNGs and examples of Monte Carlo method.
sort: 7
---
# Random Number Generators and Monte Carlo Method

* * *

## Learning objectives

*   Understand the properties of random number generators and what properties are desirable in a random number generator
*   Give examples of problems where you would use Monte Carlo
*   Characterize the error of Monte Carlo

## Random Number Generators

**_Random Number Generators (RNG)_** are algorithms or methods that can be used to generate a sequence of numbers that cannot be reasonably predicted. There are usually two principal methods for generating random numbers: **_truly-random method_** and **_pseudorandom method_**. Truly-random methods generate numbers according to some random physical phenomenon. For instance, rolling a fair die will generate truly random numbers between 1 and 6\. Other example sources include atmospheric noise and thermal noise. Pseudorandom methods generate numbers using computational algorithms that produce sequences of apparently random results, which are in fact predictable and reproducible.

When using a pseudorandom method, because only finite number of numbers can be represented in computer, any generated sequence must eventually repeat. The **_period_** of a pseudorandom number generator is defined as the maximum length of the repetition-free prefix of the sequence.

### Properties of Random Number Generators

A random number generator has the following properties:

*   Random pattern: passes statistical tests of randomness
*   Long period: goes as long as possible before repeating
*   Efficiency: executes rapidly and requires little storage
*   Repeatability: produces same sequence if started with same initial conditions
*   Portability: runs on different kinds of computers and is capable of producing same sequence on each

### Linear Congruential Generator

A **_linear congruential generator_** (LCG) is pseudorandom number generator of the form:

<div>$$ x_0 = \text{seed} $$</div>

<div>$$ x_{n+1} = (a x_{n} + c) (\text{mod} \phantom{x} M) $$</div>

where <span>\\(a\\) (the multiplier)</span> and <span>\\(c\\) (the increment)</span> are given integers and <span>\\(x_0\\)</span> is called the **_seed_**. The period of an LCG cannot exceed <span>\\(M\\) (the modulus)</span>. The quality depends on both <span>\\(a\\)</span> and <span>\\(c\\)</span>, and the period may be less than <span>\\(M\\)</span> depending on the values of <span>\\(a\\)</span> and <span>\\(c\\)</span>.

### Example of an LCG

Below is the python code for an LCG that generates the numbers \\(1,3,7,5,1,3,7,5,\dots\\) given an initial seed of <span>\\(1\\)</span>.

```python
def lcg_gen_next(modulus, a, c, xk):
  xk_p1 = (a * xk + c) % modulus
  return xk_p1

x = 1
M = 10
a = 2
c = 1
for i in range(100):
  print(x)
  x = lcg_gen_next(M, a, c, x)
```
## Random Variables
A **_Random Variable_** <span>\\(X\\)</span> can be thought of as a function that maps the outcome of unpredictable (random) processes to numerical quantities.

Examples:
- How much rain are we getting tomorrow?
- Will my buttered bread land face-down?

We don’t have an exact number to represent these random processes, 
but we can get something that represents the average case.

### Discrete Random Variables
Each **_Discrete Random Variable_** <span>\\(X\\)</span> can take a discrete value, \\(x_i\\) with probability \\(p_i\\) for \\(i = 1,...m\\) and \\(\Sigma_{i=1}^m p_i = 1\\). 
#### Coin toss example
Consider a random variable \\(X\\) which is the result of a coin toss that can be heads or tails. 
<div> $$ X=1\text{: toss is heads} $$ </div>
<div> $$ X=0\text{: toss is tails} $$ </div>
For each individual toss, \\(x_i\\) is \\(0\\) or \\(1\\) and each \\(x_i\\) has probability \\(p_i=0.5\\).

The **expected value** of a discrete random variable is defined as:
<div>$$ E(X) = \Sigma_{i=1}^m p_i x_i $$ </div>
So, for a coin toss:
<div>$$ E(X) = 1 * 0.5 + 0 * 0.5 = 0.5$$ </div>
Now, suppose we toss a “fair” coin 1000 times, and record the number of times we get heads. The recorded number would likely land close to the expected value \\(0.5\\).
If we run this \\(1000\\) coin toss experiment \\(N\\) times (let’s say \\(N=100\\)), the results will look like a normal distribution, with the majority of the results close to \\(0.5\\).
## Monte Carlo

**_Monte Carlo methods_** are algorithms that rely on repeated random sampling to approximate a desired quantity. Monte Carlo methods are typically used in modeling the following types of problems:

*   Nondeterministic processes
*   Complicated deterministic systems and deterministic problems with high dimensionality (e.g., Monte Carlo integration)

### Convergence/Error

Consider using Monte Carlo to estimate an integral \\(I = \int_a^b f(x) dx\\). Let <span>\\(X\\)</span> be a uniformly distributed random variable on <span>\\([a, b]\\)</span>. Then, \\(I = (b-a) \mathbb{E}[f(X)]\\). Using Monte Carlo with <span>\\(n\\)</span> samples, our estimate of the expected value is:

<div>\[S_n = \frac{1}{n} \sum_i^n f(X_i)\]</div>

so the approximate value for the integral is:
<div>\[ I_n = (b-a) \frac{1}{n} \sum_i^n f(X_i) \]</div>

By the law of large numbers, as \\(n \to \infty\\), the sample average <span>\\(S_n\\)</span> will converge to the expected value \\(\mathbb{E}[f(X)]\\). So, as \\(n \to \infty\\), \\(I_n \to \int_a^b f(x) dx\\).

According to central limit theorem, as \\(n \to \infty\\),
<div>\[ \sqrt{n} (S_n - \mu) \to N(0, \sigma^2) \]</div>
where \\(N(0, \sigma^2)\\) is a normal distribution; \\(\mu = \mathbb{E}[f(X)]\\) and \\(\sigma^2 = Var[X]\\).

Let <span>\\(Z\\)</span> be a random variable with normal distribution \\(N(0, \sigma^2)\\), then the error of Monte Carlo estimate, \\(err = S_n - \mu\\), can be written as
<div>\[ err \to \frac{1}{\sqrt{n}} Z \]</div>
when \\(n \to \infty\\).

Therefore, the asymptotic behavior of the Monte Carlo method is \\(\mathcal{O}(\frac{1}{\sqrt{n}})\\), where <span>\\(n\\)</span> is the number of samples.

### Example: Applying Monte Carlo

One of the most common applications of Monte Carlo is to approximate the definite integral of a complicated function, often in higher dimensions where other numerical integration techniques are extremely costly. Below is the python code for approximating the intergral of a function <span>\\(f(x,y)\\)</span> over the domain \\([x_{min}, x_{max}] \times [y_{min}, y_{max}]\\):

```python
import random

# function definition goes here
def f(x, y):
  # return function value

# set x_min, x_max, y_min and y_max for integral interval
total = 0.0

# n is the number of points used in Monte Carlo integration
for i in range(n):
  x = random.uniform(x_min, x_max)
  y = random.uniform(y_min, y_max)
  total += f(x, y)

# estimated integral value
est = (1.0/n * total)*((x_max-x_min)*(y_max-y_min))
```

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-7-random-monte-carlo.html)

## ChangeLog
*   2024-02-07 Bhargav Chandaka [bhargav9@illinois.edu](mailto:bhargav9@illinois.edu): major reorganziation to match up with content in slides/videos
*   2018-01-25 Erin Carrier [ecarrie2@illinois.edu](mailto:ecarrie2@illinois.edu): minor fixes throughout, adds review questions
*   2018-01-25 Yu Meng [yumeng5@illinois.edu](mailto:yumeng5@illinois.edu): first complete draft
*   2018-01-17 Erin Carrier [ecarrie2@illinois.edu](mailto:ecarrie2@illinois.edu): outline
