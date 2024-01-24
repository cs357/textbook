#!/usr/bin/env python

# Plots a cubic function and its root

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def f(x):
    return x**3 - x - 1

def fprime(x):
    return 3 * x**2 - 1

def fprime2(x):
    return 6 * x

x = np.linspace(-2, 2)
y = f(x)

root = opt.newton(f, 1.5, fprime=fprime, fprime2=fprime2)

plt.plot(x, y)
plt.plot(root, 0, 'ro')

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

plt.savefig('cubic.pdf')

