---
title: Markov chains
description: An application of eigenvalues
sort: 13
---
# Graphs and Markov chains

* * *

## Learning Objectives

*   Express a graph as a sparse matrix.
*   Identify the performance benefits of a sparse matrix.

## Graphs

#### Undirected Graphs:

The following is an example of an undirected graph:

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/undirected_graph.png" width=300/> </div>

The adjacency matrix, <span>\\({\bf A}\\)</span>, for undirected graphs is _always_ symmetric and is defined as:

$$ a_{ij} = \begin{cases} 1 \quad \mathrm{if} \ (\mathrm{node}_i, \mathrm{node}_j) \ \textrm{are connected} \\ 0 \quad \mathrm{otherwise} \end{cases}, $$

where <span>\\(a_{ij}\\)</span> is the <span>\\((i,j)\\)</span> element of <span>\\({\bf A}\\)</span>.
The adjacency matrix which describes the example graph above is:

<div>\[ {\bf A} = \begin{bmatrix} 0 & 1 & 1 & 1 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 1 \end{bmatrix}.\]</div>

#### Directed Graphs:

The following is an example of a directed graph:

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/directed_graph.png" width=300/> </div>


The adjacency matrix, <span>\\({\bf A}\\)</span>, for directed graphs is defined as:

<div>\[ a_{ij} = \begin{cases} 1 \quad \mathrm{if} \ \mathrm{node}_i \leftarrow \mathrm{node}_j \\ 0 \quad \mathrm{otherwise} \end{cases}, \]</div>

where <span>\\(a_{ij}\\)</span> is the <span>\\((i,j)\\)</span> element of <span>\\({\bf A}\\)</span>. The adjacency matrix which describes the example graph above is:

<div>\[ {\bf A} = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 1 \end{bmatrix}.\]</div>

#### Weighted Directed Graphs:

The following is an example of a weighted directed graph:

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/weighted_directed_graph.png" width=300/> </div>

The adjacency matrix, <span>\\({\bf A}\\)</span>, for weighted directed graphs is defined as:

<div>\[ a_{ij} = \begin{cases} w_{ij} \quad \mathrm{if} \ \mathrm{node}_i \leftarrow \mathrm{node}_j \\ 0 \quad \ \ \mathrm{otherwise} \end{cases}, \]</div>

where <span>\\(a_{ij}\\)</span> is the <span>\\((i,j)\\)</span> element of <span>\\({\bf A}\\)</span>, and <span>\\(w_{ij}\\)</span> is the link weight associated with edge connecting nodes <span>\\(i\\)</span> and <span>\\(j\\)</span>. The adjacency matrix which describes the example graph above is:

<div>\[ {\bf A} = \begin{bmatrix} 0 & 0 & 0 & 0.4 & 0 & 0 \\ 0.1 & 0.5 & 0 & 0 & 0 & 0 \\ 0.9 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1.0 & 0 & 1.0 & 0 \\ 0 & 0.5 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0.6 & 0 & 1.0 \end{bmatrix}.\]</div>

Typically, when we discuss weighted directed graphs it is in the context of transition matrices for Markov chains where the link weights across each column sum to <span>\\(1\\)</span>.

## Markov Chain

A **_Markov chain_** is a stochastic model where the probability of future (next) state depends only on the most recent (current) state. This memoryless property of a stochastic process is called **_Markov property_**. From a probability perspective, the Markov property implies that the conditional probability distribution of the future state (conditioned on both past and current states) depends only on the current state.

## Markov Matrix

A **_Markov/Transition/Stochastic matrix_** is a square matrix used to describe the transitions of a Markov chain. Each of its entries is a non-negative real number representing a probability. Based on Markov property, next state vector \\({\bf x}_{k+1}\\) is obtained by left-multiplying the Markov matrix <span>\\({\bf M}\\)</span> with the current state vector \\({\bf x}_k\\).
<div>\[ {\bf x}_{k+1} = {\bf M} {\bf x}_k \]</div>
In this course, unless specifically stated otherwise, we define the transition matrix <span>\\({\bf M}\\)</span> as a left Markov matrix where each column sums to <span>\\(1\\)</span>.

_Note_: Alternative definitions in outside resources may present <span>\\({\bf M}\\)</span> as a right markov matrix where each row of <span>\\({\bf M}\\)</span> sums to <span>\\(1\\)</span> and the next state is obtained by right-multiplying by <span>\\({\bf M}\\)</span>, i.e. \\({\bf x}_{k+1}^T = {\bf x}_k^T {\bf M}\\).

A steady state vector \\({\bf x}^*\\) is a probability vector (entries are non-negative and sum to <span>\\(1\\)</span>) that is unchanged by the operation with the Markov matrix <span>\\(M\\)</span>, i.e.
<div>\[ {\bf M} {\bf x}^* = {\bf x}^* \]</div>
Therefore, the steady state vector \\({\bf x}^*\\) is an eigenvector corresponding to the eigenvalue \\(\lambda=1\\) of matrix <span>\\({\bf M}\\)</span>. If there is more than one eigenvector with \\(\lambda=1\\), then a weighted sum of the corresponding steady state vectors will also be a steady state vector. Therefore, the steady state vector of a Markov chain may not be unique and could depend on the initial state vector.

## Markov Chain Example

Suppose we want to build a Markov Chain model for weather predicting in UIUC during summer. We observed that:

*   a sunny day is \\(60\%\\) likely to be followed by another sunny day, \\(10\%\\) likely followed by a rainy day and \\(30\%\\) likely followed by a cloudy day;
*   a rainy day is \\(40\%\\) likely to be followed by another rainy day, \\(20\%\\) likely followed by a sunny day and \\(40\%\\) likely followed by a cloudy day;
*   a cloudy day is \\(40\%\\) likely to be followed by another cloudy day, \\(30\%\\) likely followed by a rainy day and \\(30\%\\) likely followed by a sunny day.

The state diagram is shown below:

<div class="figure"> <img src="{{ site.baseurl }}/assets/img/figs/weather.png" width=500/> </div>

The Markov matrix is
<div>\[ {\bf M} = \begin{bmatrix} 0.6 & 0.2 & 0.3 \\ 0.1 & 0.4 & 0.3 \\ 0.3 & 0.4 & 0.4 \end{bmatrix}. \]</div>

If the weather on day <span>\\(0\\)</span> is known to be rainy, then
<div>\[ {\bf x}_0 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}; \]</div>
and we can determine the probability vector for day <span>\\(1\\)</span> by
<div>\[ {\bf x}_1 = {\bf M} {\bf x}_0. \]</div>
The probability distribution for the weather on day <span>\\(n\\)</span> is given by
<div>\[ {\bf x}_n = {\bf M}^{n} {\bf x}_0. \]</div>

## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-13-markov.html)

## ChangeLog

*   2018-04-01 Erin Carrier <ecarrie2@illinois.edu>: Minor reorg and formatting changes
*   2018-03-25 Yu Meng <yumeng5@illinois.edu>: adds Markov chains
*   2018-03-01 Erin Carrier <ecarrie2@illinois.edu>: adds more review questions
*   2018-01-14 Erin Carrier <ecarrie2@illinois.edu>: removes demo links
*   2017-11-02 Erin Carrier <ecarrie2@illinois.edu>: adds changelog, fix COO row index error
*   2017-10-25 Erin Carrier <ecarrie2@illinois.edu>: adds review questions, minor fixes and formatting changes
*   2017-10-25 Arun Lakshmanan <lakshma2@illinois.edu>: first complete draft
*   2017-10-16 Luke Olson <lukeo@illinois.edu>: outline
