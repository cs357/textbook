---
title: Nonlinear Least Squares
description: Add description here...
sort: 15
---
# Nonlinear Least Squares

## Non-linear Least-squares Problem vs. Linear Least-squares Problem

The above linear least-squares problem is associated with an overdetermined linear system \\(A {\bf x} \cong {\bf b}.\\) This problem is called a linear one because the fitting function we are looking for is linear in the components of \\({\bf x}\\). For example, if we are looking for a polynomial fitting function
<div>\[ f(t,{\bf x}) = x_1 + x_2t + x_3t^2 + \dotsb + x_nt^{n-1} \]</div>
to fit the data points \\(\{(t_i,y_i)|1 \le i \le m\}\\) (<span>\\(m > n\\)</span>), the problem is still a linear least-squares problem, because \\(f(t,{\bf x})\\) is linear in the components of \\({\bf x}\\) (though \\(f(t,{\bf x})\\) is nonlinear in <span>\\(t\\)</span>).

If the fitting function \\(f(t,{\bf x})\\) for data points \\(\{(t_i,y_i)|1 \le i \le m\}\\) is nonlinear in the components of \\({\bf x}\\), then the problem is a non-linear least-squares problem. For example, fitting sum of exponentials
<div>\[ f(t,{\bf x}) = x_1 e^{x_2 t} + x_2 e^{x_3 t} + \dotsb + x_{n-1} e^{x_n t} \]</div>
is a **_non-linear least-squares problem_**.


## Non-linear Least-squares Problem Set Up

Given <span>\\(m\\)</span> data points, \\(\{(t_1,y_1),(t_2,y_2),\dots,(t_m,y_m)\}\\), we want to find a curve \\(f(t,{\bf x})\\) that best fits the data points. Mathematically, we are finding \\({\bf x}\\) such that the squared Euclidean norm of the residual vector
<div>\[ \|{\bf r}({\bf x})\|_2^2 \]</div>
is minimized, where the components of the residual vector are
<div>\[ r_i({\bf x}) = y_i - f(t_i,{\bf x}). \]</div>

Equivalently, we want to minimize
<div>\[ \phi({\bf x}) = \frac{1}{2} {\bf r}^T({\bf x}){\bf r}({\bf x}). \]</div>

{% assign opt_page = site.references | where: "index", "14" | first %}

To solve this optimization problem, we can use steepest descent, see [Reference Page: Optimization]({{ site.baseurl }}{{ opt_page.url }}). We compute the gradient vector
<div>\[ \nabla \phi({\bf x}) = J^T({\bf x}){\bf r}({\bf x}) ,\]</div>
where \\(J({\bf x})\\) is the Jacobian of \\({\bf r}({\bf x})\\).

Then, we use steepest descent as usual with this gradient vector and our objective function, \\(\phi({\bf x})\\), iterating until we converge to a solution.


## Example of a Non-linear Least Squares Problem

Consider fitting <span>\\(m\\)</span> data points, \\((t_0, y_0), (t_1, y_1), \dots, (t_{m-1}, y_{m-1})\\), with the curve
<div>\[f(t, \mathbf{x}) = x_0 + x_1 e^{-x_2 t}.\]</div>

The components of the residual are given by:
<div>\[r_i(\mathbf{x}) = y_i - (x_0 + x_1 e^{-x_2 t_i}).\]</div>

The gradient of \\(\phi(\mathbf{x})\\) is given by:
<div>\[\begin{bmatrix} -1 & -1 & \dots & -1 \\ -e^{-x_2 t_0} & -e^{-x_2 t_1} & \dots & -e^{-x_2 t_{m-1}} \\ x_1 t_0 e^{-x_2 t_0} & x_1 t_1 e^{-x_2 t_1} & \dots & x_1 t_{m-1} e^{-x_2 t_{m-1}} \end{bmatrix} \begin{bmatrix} y_0 - x_0 - x_1 e^{-x_2 t_0} \\ \vdots \\ y_{m-1} - x_0 - x_1 e^{-x_2 t_{m-1}} \end{bmatrix}. \]</div>

With steepest descent, we would use a line search along the direction of the negative gradient to find the overall step and use this to find the next iterate.

## Review Questions

11.  For a given model and given data, what is the residual vector for a nonlinear least squares problem?
12.  How do we solve a nonlinear least squares problem? What do we minimize when solving a nonlinear least squares problem?
13.  Consider solving a nonlinear least squares problem using gradient descent. For a given model, how do you compute the direction of the step?
