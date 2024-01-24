---
title: Finite Difference Methods
description: Numerical approximation of derivatives
sort: 14
---
# Finite Difference Methods

* * *

## Learning Objectives

* Approximate derivatives using the Finite Difference Method

## Finite Difference Approximation

For a differentiable function \\(f:\mathbb{R} \rightarrow \mathbb{R}\\), the derivative is defined as

<div>\[f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h} \]</div>

Let's consider the forward finite difference approximation to the first derivative as

<div>\[f'(x) \approx \frac{f(x+h)-f(x)}{h} \]</div>

where <span>\\(h\\)</span> is often called a "perturbation", i.e., a "small" change to the variable <span>\\(x\\)</span> (small when compared to the magnitude of <span>\\(x\\)</span>). By the Taylor's theorem, we can write

<div>\[f(x+h) = f(x) + f'(x)\, h + f''(\xi)\, \frac{h^2}{2} \]</div>

for some \\(\xi \in [x,x+h]\\). Rearranging the above we get

<div>\[f'(x) = \frac{f(x+h)-f(x)}{h} - f''(\xi)\, \frac{h}{2} \]</div>

Therefore, the truncation error of the finite difference approximation is bounded by \\(M\,h/2\\), where <span>\\(M\\)</span> is a bound on $$ \vert f''(\xi) \vert $$ for \\(\xi\\) near <span>\\(x\\)</span>.



Using a similar approach, we can summarize the following finite difference approximations:

#### Forward Finite Difference Method

In addition to the computation of \\(f(x)\\), this method requires one function evaluation for a given perturbation, and has truncation order \\(O(h) \\).

$$f'(x) = \frac{f(x+h)-f(x)}{h}$$


#### Backward Finite Difference Method

In addition to the computation of \\(f(x)\\), this method requires one function evaluation for a given perturbation, and has truncation order \\(O(h) \\).

$$f'(x) = \frac{f(x)-f(x-h)}{h}$$

#### Central Finite Difference Method

This method requires two function evaluations for a given perturbation (\\(f(x+h)\\) and \\(f(x-h)\\) ), and has truncation order \\(O(h^2) \\).

$$f'(x) = \frac{f(x+h)-f(x-h)}{2h}$$


*Reference text: "Scientific Computing: an introductory survey" by Michael Heath*


## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-6-taylor.html)

## ChangeLog

*   2021-01-20 Mariana Silva [mfsilva@illinois.edu](mailto:mfsilva@illinois.edu): Moved FD content from Taylor to this new section
