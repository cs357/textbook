---
permalink: /notes/python
title: Python
description: An elementary introduction to Python.
author:
- Mariana Silva
- CS 357 Course Staff
---

# Introduction to Python

## Learning Objectives

*   Get familiar with the language of Python 3
*   Understand the difference between mutable and immutable objects
*   Remember, the most efficient way to study a language is to practice!
*   Google is a great resource for any programming language


## Mutable and Immutable Objects

In Python, objects ae divided into **mutable** and **immutable**. Mutable objects can be modified after they are first created, including lists, dictionaries, numpy arrays, etc. Immutable objects cannot be modified once they are created, including tuples, strings, floats, etc. 

For example,

```python
# A successful attemp to replace the first entry of the list

myList = [3, 5, 7]
myList[0] = 1



# The following two attempts will result in a TypeError: object does not suppoer item assignment

myTuple = (3, 5, 7)
myTuple[0] = 1

myString = "357"
myString[0] = '1'
```


## Dictionary

Dictionary is an important and useful structure in Python. I wil provide an example for useful operations on a dictionary:

```python
# definition of a dictionary
myDict = {
    "number": 357,
    "major": "computer science",
    "credit hours": 3
}

# dictionaries are mutable, thus can be modified, either modifying an existing value or creating a new key
myDict["major"] = "math"
myDict["level"] = "undergrad"

# looping through the keys or the values over the dictionary
for x in myDict:
  print(x)
  print(myDict[x])
for x in myDict.values():
  print(x)
  
# there are multiple ways to remove an existing entry, here is one example
myDict.pop("level")

# copy an existing dictionary into a new reference
anotherDict = myDict.copy()
```
Here is the [official documentation](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) for dictionary and some other useful data structures. 


## User Defined Functions

Sometimes to define a funstion si useful. The keyword "def" followed by a list of parameters is used to define functions. Here is a basic example of function definition:

```python
# a function to return the area of a square
def area(length):
  return length ** 2

print(area(2))
```
Here is the [official documentation](https://docs.python.org/3.8/tutorial/controlflow.html#defining-functions) for function definition. 


## External Links
Here are some links to some packages we will be using in CS 357.

- Documentation for [numpy](https://numpy.org/doc/stable/).

- Docementation for [scipy](https://docs.scipy.org/doc/scipy/reference/).

- Documentation for [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot).


## Review Questions

- See this [review link](/cs357/fa2020/reviews/rev-2-python.html)

## Changelog
* 2024-01-12 Dev Singh dsingh14@illinois.edu: Port to 