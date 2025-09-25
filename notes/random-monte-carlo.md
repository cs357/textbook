---
title: Random Number Generators and Monte Carlo Method
description: Properties of RNGs and examples of Monte Carlo method.
sort: 7
author:
  - CS 357 Course Staff
changelog:
  - 
    name: Bhargav Chandaka
    netid: bhargav9
    date: 2024-02-17
    message: major reorganziation to match up with content in slides/videos
  - 
    name: Erin Carrier
    netid: ecarrie2
    date: 2018-01-25
    message: minor fixes throughout, adds review questions
  - 
    name: Yu Meng
    netid: yumeng5
    date: 2018-01-25
    message: first complete draft
  - name: Erin Carrier
    netid: ecarrie2
    date: 2018-01-17
    message: outline

---

## Learning Objectives

*   Understand the properties of random number generators and what properties are desirable in a random number generator.
*   Give examples of problems where you would use Monte Carlo.
*   Characterize the error of Monte Carlo.

## Random Number Generators

**_Random Number Generators (RNG)_** are algorithms or methods that can be used to generate a sequence of numbers that cannot be reasonably predicted. There are usually two principal methods for generating random numbers: **_truly-random methods_** and **_pseudorandom methods_**. Truly-random methods generate numbers according to some random physical phenomenon. For instance, rolling a fair die will generate truly random numbers between 1 and 6\. Other example sources include atmospheric noise and thermal noise. Pseudorandom methods generate numbers using computational algorithms that produce sequences of apparently random results, which are in fact predictable and reproducible.

When using a pseudorandom method, any generated sequence must eventually repeat since only a finite quantity of numbers can be represented in computer. The **_period_** of a pseudorandom number generator is defined as the maximum length of the repetition-free prefix of the sequence.

### Properties of Random Number Generators

A random number generator has the following properties:

*   *Random pattern*: passes statistical tests of randomness.
*   *Efficiency*: executes rapidly and requires little storage.
*   *Long period*: goes as long as possible before repeating.
*   *Repeatability*: produces same sequence if started with same initial conditions.
*   *Portability*: runs on different kinds of computers and is capable of producing same sequence on each.

### Linear Congruential Generator

A **_linear congruential generator_** (LCG) is pseudorandom number generator of the form:

<div>$$ x_0 = \text{seed} $$</div>

<div>$$ x_{n+1} = (a x_{n} + c) (\text{mod} \phantom{x} M) $$</div>

where <span>\\(a\\) (the multiplier)</span> and <span>\\(c\\) (the increment)</span> are given integers, and <span>\\(x_0\\)</span> is called the **_seed_**. The period of an LCG cannot exceed <span>\\(M\\) (the modulus)</span>. The period may be less than <span>\\(M\\)</span> depending on the values of <span>\\(a\\)</span> and <span>\\(c\\)</span>. The quality depends on both <span>\\(a\\)</span> and <span>\\(c\\)</span>. 

### Example of a LCG

Below is the Python code for an example LCG that generates the numbers \\(1,3,7,5,1,3,7,5,\dots\\) given an initial seed of <span>\\(1\\)</span>.
To follow the pattern, we double the previous number, add \\(1\\), and mod by 10, so \\(a=2\\), \\(c=1\\), and \\(M=10\\). 
```python
def lcg_gen_next(modulus: int, a: int, c: int, xk: int) -> int:
  """ Uses an LCG to generate a pseudo-random number.
  
  Args:
    modulus (int): the period of the LCG
    a (int): the multiplier
    c (int): the increment
    xk (int): the previously generated random number(or the seed)

  Returns:
    int: the next pseudo-random number, xk_p1 
  """ 
  xk_p1 = (a * xk + c) % modulus
  return xk_p1

x = 1 # initial seed
M = 10 
a = 2
c = 1
for i in range(100):
  print(x)
  x = lcg_gen_next(M, a, c, x)
```
## Random Variables
A **_random variable_** <span>\\(X\\)</span> can be thought of as a function that maps the outcome of unpredictable (random) processes to numerical quantities. If you have previous statistics experience, you are likely familiar with this concept.

Examples of random variables include:
- How much rain are we getting tomorrow?
- Will my buttered bread land face-down?

We don’t have an exact number to represent these random processes, but we can approximate the long-run outcome of random variables through the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers).

### Discrete Random Variables
Each **_discrete random variable_** <span>\\(X\\)</span> can take a discrete value, \\(x_i\\) with probability \\(p_i\\) for \\(i = 1,...m\\) and \\(\Sigma_{i=1}^m p_i = 1\\) (i.e. the possible values are countable). This is in contrast to continuous random variables, where the possible values are uncountably infinite.

#### Expected Value of a Discrete Random Variable
The **expected value** of a discrete random variable, $$E(x)$$, is defined as $$\Sigma_{i=1}^m p_i x_i$$. This concept can be thought of as the "average outcome". 

### Example of a Discrete Random Variable: Coin Toss
Consider a random variable \\(X\\) which is the result of a coin toss that can be heads or tails. 
<div> $$ X=1\text{: toss is heads} $$ </div>
<div> $$ X=0\text{: toss is tails} $$ </div>
For each individual toss, \\(x_i\\) is \\(0\\) or \\(1\\) and each \\(x_i\\) has probability \\(p_i=0.5\\).

What is the expected value for this coin toss?
<details> 
<summary><strong>Answer</strong></summary>
For a coin toss, the expectation is
<div>$$ E(X) = 1 \cdot 0.5 + 0 \cdot 0.5 = 0.5$$ </div>
</details>
Now, suppose we toss a “fair” coin 1000 times, and record the number of times we get heads. What would the distribution look like if we run this \\(1000\\) coin toss experiment \\(N\\) times (let’s say \\(N=100\\)) ? 
<details>
<summary><strong>Answer</strong></summary>
The recorded number for each 1000 coin toss experiment would likely land close to the expected value \(0.5\). The results of running the experiment \(N\) times would look will look like a normal distribution, with the majority of the results close to \(0.5\). This is also known as the law of large numbers (when N is large).
</details>
## Monte Carlo

**_Monte Carlo methods_** are algorithms that rely on repeated random sampling to approximate a desired quantity. Monte Carlo methods are typically used in modeling the following types of problems:

*   Nondeterministic processes,
*   Complicated deterministic systems and deterministic problems with high dimensionality (e.g. integration of non-trivial functions).

One of the most common applications of Monte Carlo is to approximate the area/volume for a given shape. For example, to approximate the area of a circle, we can first uniformly sample a large number of points in a square region around the circle. Then, we check which points are inside the circle to get a percentage of points in the circle, \\(p\\). Finally, we approximate the circle's area by multiplying \\(p\\) by the square's area. Monte Carlo works in this scenario because:
1. **Uniform random sampling** means that we ensure points are scattered throughout region and cover it well
2. Based on the **law of large numbers**, as we increase the number of samples, the average area value will converge to the true area value. So, using a large number of samples improves our estimate. 


### Integration with Monte Carlo

Consider using Monte Carlo to estimate an integral \\(I = \int_a^b f(x) dx\\). Let <span>\\(X\\)</span> be a uniformly distributed random variable on <span>\\([a, b]\\)</span>. Then, \\(I = (b-a) \mathbb{E}[f(X)]\\). Using Monte Carlo with <span>\\(n\\)</span> samples, our estimate of the expected value is:

<div>\[S_n = \frac{1}{n} \sum_i^n f(X_i)\]</div>

so the approximate value for the integral is:
<div>\[ I_n = (b-a) \frac{1}{n} \sum_i^n f(X_i) \]</div>

By the law of large numbers, as \\(n \to \infty\\), the sample average <span>\\(S_n\\)</span> will converge to the expected value \\(\mathbb{E}[f(X)]\\). So, as \\(n \to \infty\\), \\(I_n \to \int_a^b f(x) dx\\).

According to central limit theorem, as \\(n \to \infty\\),
<div>\[ \sqrt{n} (S_n - \mu) \to N(0, \sigma^2) \]</div>
where \\(N(0, \sigma^2)\\) is a normal distribution; \\(\mu = \mathbb{E}[f(X)]\\) and \\(\sigma^2 = Var[X]\\).

### Error/Convergence

Let <span>\\(Z\\)</span> be a random variable with normal distribution \\(N(0, \sigma^2)\\). Then the error of Monte Carlo estimate, \\(err = S_n - \mu\\), can be written as
<div>\[ err \to \frac{1}{\sqrt{n}} \sigma \]</div>
when \\(n \to \infty\\).

Therefore, the asymptotic behavior of the Monte Carlo method is \\(\mathcal{O}(\frac{1}{\sqrt{n}})\\), where <span>\\(n\\)</span> is the number of samples.


### Example: Applying Monte Carlo
 We can use Monte Carlo to efficiently approximate the definite integral of a complicated function, which is especially useful in higher dimensions when other numerical integration techniques are extremely costly.  Below is the Python code for approximating the intergral of a function <span>\\(f(x,y) = x+y\\)</span> over the domain \\([x_{min}, x_{max}] \times [y_{min}, y_{max}]\\):

```python
import random

def f(x: float, y: float) -> float:
  """ Returns the value for function *f* at point (x, y).
  
  Args:
    x: float: the point x.
    y: float: the point y.

  Returns:
    float: f(x, y)
  """ 
  return x + y

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
1. What is a pseudo-random number generator?
2. What are properties of good random number generators?
3. What are advantages/disadvantages of pseudorandom number generators in comparison to using truly random numbers?
4. What is a linear congruential generator (LCG)?
5. What is a seed for a random number generator?
6. Do random number generators repeat? Are they reproducible?
7. What are Monte Carlo methods and how are they used?
8. For Monte Carlo, how does the error behave in relation to the number of sampling points?
9. Given a computed value from Monte Carlo and a sampling error, what sampling error could you expect for a different number of samples?
10. For a small example problem, use Monte Carlo to estimate the area of a certian domain.
11. For a small example problem, use Monte Carlo to estimate the integral of a function.
