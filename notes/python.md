---
title: Python
description: An elementary introduction to Python.
sort: 2
author:
  - CS 357 Course Staff
changelog:
  - 
    name: Dev Singh
    netid: dsingh14
    date: 2024-03-17
    message: modified notes content for clarity
  - 
    name: Arnav Aggarwal
    netid: arnava4
    date: 2024-03-16
    message: aligned notes with slides and added additional examples
---

# Python

* * *

## Learning Objectives

*   Get familiar with the Python 3 syntax
*   Understand the difference between mutable and immutable objects
*   Understand the purpose of NumPy and learn about common operations with NumPy datatypes.


## Notes
1. Remember, the most efficient way to learn a programming language is to practice!

2. Google is a great resource for any programming language.

## Types
  Just like any other language, variables can store data of different types in Python, such as `int`, `float`, or `str`. We can use the `type()` function to find the type of an expression.

### Example

```python
a = 2
b = 3.0
c = a + b
d = 2 * a
```

What are the correct types for variables `c` and `d`?

$$
\begin{flalign} 

 & \text{A) } \text{c is a float, d is a float} \\ 

 & \text{B) } \text{c is a float, d is an int} \\ 

 & \text{C) } \text{c is an int, d is an int} \\ 

 & \text{D) } \text{c is an int, d is a float} \\ 

\end{flalign}
$$

<details>
    <summary><strong>Answer</strong></summary>

<br>
<b>B.</b>
<br>
<br> 

In Python, any operation between an int and a float will result in a float. Hence, c is a float. d is an int because we did a multiplication operation on two ints. 

</details>


## Names and Values
Consider the following example:

```python
a = [1, 2, 3]
b = a
```

In this case, `a` and `b` are not separate entities, even though they are different variables. The list ```[1, 2, 3]``` is an object, and both variable names a and b are bounded to the same list. 

### Modifying an Object
Now let's consider another example:

```python
a = [1, 2, 3]
b = a
b.append(4)
```

```b.append(4)``` modifies the object list, such that the list is now ```[1, 2, 3, 4]```. We know that b is now  ```[1, 2, 3, 4]```, but what happens to the variable name a?

<details>
    <summary><strong>Answer</strong></summary>

Because a and b are bounded to the same list they will have the same values once the list is modified. 

</details>

### Get the "id" for an object

Let's see how we can use a built-in Python function to see whether a and b point to the same object or not. 

```python
# Since "a" and "b" are bounded to the same list object, they will have the same "id"
print(id(a), id(b))

# The "is" keyword is an additional check to see if both variable names have the same "id"
a is b
```

### Memory Management
To review the way Python manages memory, let's look at the following examples:

```python
# The following code will print "IS  False" and "EQUAL True" because a and b are bounded to two different list objects.
# Hence, they don't have the same id but the contents of their data are the same. 

a = [1, 2, 3]
b = [1, 2, 3]
print("IS  ", a is b)
print("EQUAL", a == b)


# The following code is the case we've seen before. 
# a and b are bounded to the same list object, and hence they'll have the same id. 

a = [1, 2, 3]
b = a
```


### Mutable and Immutable Objects

In Python, objects are divided into **mutable** and **immutable**. Mutable objects can be modified after they are first created, including lists, dictionaries, NumPy arrays, etc. Immutable objects cannot be modified once they are created, including tuples, strings, floats, etc. 

For example,

```python
# A successful attempt to replace the first entry of the list

myList = [3, 5, 7]
myList[0] = 1



# The following two attempts will result in a TypeError: object does not suppoer item assignment

myTuple = (3, 5, 7)
myTuple[0] = 1

myString = "357"
myString[0] = '1'
```

#### List

A list is an example of a mutable object. Let's take a look at how it behaves in different situations:

```python
# Again, the case we've seen before: a and b are bounded to the same list object

a = [1, 2, 3]
b = a


# Here, our variable a gets reassigned to a new object, but b is still bounded to the initial object

a = a + [4]
print(b)
print(a)
a is b      # evaluates to False


# In this case, the object list is modified, however, a and b remain bounded to the object

a += [4]
print(b)
print(a)
a is b      # evaluates to True

```

### Example

Which of the following code snippets result in ```print(a==c) -> True```:

A:
```python
a = ['hello','goodbye']
b = 'hey'
a.append(b)
c = a + [b]
```

B:
```python
a = ['hello','goodbye']
b = 'hey'
c = a + [b]
a += b
```

C:
```python
a = ['hello','goodbye']
b = 'hey'
c = a + [b]
a.append(b)
```

<details markdown="1">

<summary><strong>Answer</strong></summary>

<br />
**C**

A is incorrect because `c` ends up having two "hey" elements in its list. B is incorrect because `a += b` adds each character of "hey" as its own element. Hence, C must be the correct choice. 

</details>

## Objects and Naming

### Advanced Naming
Let's make some objects and see what happens in this next snippet of code:

```python
fruit = 'apple'

lunch = []
lunch.append(fruit)       # lunch = ['apple']

dinner = lunch            # dinner = ['apple'], lunch = ['apple']
dinner.append('fish')     # dinner = ['apple', 'fish'], lunch = ['apple', 'fish']

fruit = 'pear'            

meals = [fruit, lunch, dinner] 
print(meals)              # meals = ['pear', ['apple', 'fish'], ['apple', 'fish']]
```

### Example

What is the correct output for the following code snippet?

```python
John = 'computer_science'
Tim = John
Tim += ', math'
Anna = ['electrical']
Julie = Anna
Julie += ['physics']
print(John, Anna)
```

A:
```computer_science, math ['electrical', 'physics']```

B:
```computer_science, math ['electrical']```

C:
```computer_science ['electrical', 'physics']```

D:
```computer_science ['electrical']```

<details markdown="1">
    
<summary><strong>Answer</strong></summary>

<br>

**C**

In the above code snippet, John and Tim refer to separate objects since strings are immutable in Python, and so the variable John is equal to `'computer_science'`. Lists are mutable, so Julie and Anna are equal to the same list: `['electrical', 'physics']`. Hence, C is the correct choice. 

</details>


## Indexing

Indexing is important to iterate through for loops in a variety of ways in Python. Given a list `a`, the formatting for indexing follows this standard: ```a[i:j:k]```. Here, `i` is the starting index of the iteration, `j` is the stopping index (exclusive) of the iteration, and `k` is the step size.

### Example
```python
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
a[1::2][::-1]
```

What is the output for the command line above?

$$
\begin{flalign} 

 & \text{A) } [1, 3, 5, 7, 9] \\ 

 & \text{B) } [1, 3] \\ 

 & \text{C) } [3, 1] \\ 

 & \text{D) } [9, 7] \\

 & \text{E) } [9, 7, 5, 3, 1] \\ &&

\end{flalign}
$$

<details>
    <summary><strong>Answer</strong></summary>

<br>

\(\bf E\)

<br>
<br> 

The first step is to get the resulting array of a[1::2]. The start index is 1 in a zero-indexed array, and the step size is 1 until the end of the array (since no stop index was given here). Hence, the resulting array is [1, 3, 5, 7, 9], which we'll call array b for now. 
<br>
<br>
The final step is to get the resulting array of b[::-1]. There is not start or stop index given, but the step size is -1. This means that we simply need to reverse the array b, as a negative step size means we start the iteration from the end of the array rather than the start. Hence, the final array will be [9, 7, 5, 3, 1], which equals answer choice E. 

</details>

##  Control Flow

Control flow includes iterating over list-like objects with for/while loops, using if/else blocks to perform logic for certain conditions, and using break and continue statements to control when you exit a for loop or move onto the next iteration of a loop, respectively. Let's see a simple example of this:

```python
# builds a list of the squares of integers below 50 divisible by 7

mylist = []
for i in range(50):

  if i % 7 == 0:
    mylist.append(i**2)


# does this using something called list comprehension, a concise and easy way to write the code block above

mylist = [i**2 for i in range(50) if i % 7 == 0]
print(mylist)
```


## Functions


### Defining Functions
Sometimes it's useful to define a function. This is done by using the keyword "def" followed by a list of parameters. Here is a basic example of a function definition:

```python
# a function to return the area of a square
def area(length):
  return length ** 2

print(area(2))
```
Here is the [official documentation](https://docs.python.org/3.8/tutorial/controlflow.html#defining-functions) for function definition. 


### Function Scope

It's important to understand the idea of scope. A variable created inside a function can only be used inside that function, because it has **local scope**. A variable created in the main body of the Python code is a global variable and hence has **global scope**. This variable has no restrictions and can be used/accessed anywhere. Again, the best way to learn this is by looking at an example:

```python
def add_minor(person):
  person.append('math')

def switch_majors(person):
  person = ['physics']      # here, the person variable has a local scope, meaning its value/update only happens within the function, and is "ignored" outside of it
  person.append('economics')

# all variables created out here have a global scope

John = ['computer_science']
Tim = John
add_minor(Tim)
switch_majors(John)
print(John, Tim)
```

### Example

Which code snippet does not modify the variables?

A
```python
a = [3, 4]
b = [6, 7]

def do_stuff(a, b):
  return (a.append(5), b.append(8))

do_stuff(a, b)
```

B
```python
a = 3
b = 5

def do_stuff(a, b):
  a += 1
  b += 2

do_stuff(a, b)
```

C
```python
a = [3, 4]
b = [6, 7]

def do_stuff(a, b):
  a += [5]
  b += [8]

do_stuff(a, b)
```

<details markdown="1">

<summary><strong>Answer</strong></summary>

<br>

**B**

B is the correct answer because `ints` in Python are immutable. This means that incrementing `a` by 1 and `b` by 2 inside the function just creates new objects for both a and b. These new objects will be restricted to the scope of this function, and as a result the variables will not be modified. 
<br>
However, lists are mutable objects, and appending values to them does not create new objects. The function simply updates the variables that already have global scope, and so these variables are modified. 
</details>


## Dictionary

A dictionary is an important and useful structure in Python. A dictionary is made up of key-value pairs, where the value is extracted using the key. You will be using dictionaries a lot in this class for later chapters. Here are some examples on how to use a dictionary:

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
Here is the [official documentation](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) for dictionaries and some other useful data structures. 


## Type Annotations

Python is a dynamically typed language, meaning that the variable type is determined at runtime. As you've likely noticed, this is why we have the luxury to declare and initialize variables without assigning a type to the variable. This wouldn't be allowed in languages like C++ or Java. 

<strong> Type Annotations</strong>, however, provide a way for developers to specify the type of a variable, function parameter, or function result. While it doesn't really improve performance, it helps developers get a better understanding of the type of data they're dealing with, whether that be in the form of variables or inputs and outputs of a function call. Here's an example of what a type-annotated variable/function definition might look like: 

```python
# simple variable declaration (the type we're used to)
a = 5

# type annotated variable declaration
a: int = 5


# simple function definition (the type we're used to)
def add_two_numbers(a, b):
  return a + b

# type annotated function definition (both inputs and output)
def add_two_numbers(a: int, b: int) -> int:
  return a + b
```

## NumPy

NumPy is a Python library used for numerical computing. Developed almost entirely in C and C++, it is highly performant and can easily support operations on large amounts of data. This section will serve as a brief intro to some common NumPy tools you'll be expected to use throughout the class. While this section will explain some of these common tools, the [NumPy documentation](https://NumPy.org/doc/) goes into much greater detail and is always the go-to resource whenever unsure how to use a specific function.

### NumPy Arrays

NumPy arrays are generally preferred over lists because you can conduct highly performant operations on them to accomplish any task. This is specifically important in a class like Numerical Methods. The following are ways you can initialize different types of NumPy arrays: 

```python
import numpy as np
# creates a 2d array of zeros (conceptually a matrix) that has shape 2 x 2
np.zeros((2, 2))

# creates a 2d array of ones (conceptually a matrix) that has shape 2 x 2
np.ones((2, 2))

# creates an array of numbers that are evenly spaced between 2 and 3 (4 numbers in this case)
np.linspace(2, 3, 4)

# creates a 2d array of random numbers between 0 and 1 that has shape 2 x 2
np.random.rand(2, 2)

# creates an empty 2d array (conceptually a matrix) that has shape 2 x 2
np.empty((2, 2))
```

We can find out additional information about the NumPy array and do more with them with the following functions:

```python
import numpy as np
a = np.zeros((2, 2))

# we can get the shape of our NumPy array with the following function
print(a.shape)        # will return (2, 2)

# this will give us the data type of the array elements
print(a.dtype)        # returns float

# If we want to convert each element in the array to an int instead of float, we can perform the following operation: 
a = a.astype(int)
print(a.dtype)        # returns int

# this creates a deep copy of the array, so changing one won't affect the other
b = a.copy()
```

### Indexing and Slicing

We can use indexing and slicing on NumPy arrays in order to extract specific information from arrays/matrices. This will be a constant part of the class, so it's better to get used to indexing/slicing now! Note that this section is similar to the above indexing section. A NumPy array a will follow the standard: ```a[i:j:k]```, where `i` is the starting index of the iteration, `j` is the stopping index (exclusive) of the iteration, and `k` is the step size.

```python
import numpy as np
a = np.array([3, 7, 9, 10, 3, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])

# basic indexing for both 1d and 2d NumPy arrays (for 2d arrays we specify both the row and col)
print(a[2])     # prints 9
print(b[0, 0])  # prints 1

# slicing examples for both 1d and 2d NumPy arrays (for 2d arrays we specify both the row and col)
print(a[1:3])     # prints [7, 9]
print(b[0:1, 2])  # prints [3]

# If we leave the row/col index empty or use a colon(:) then we're saying that we select the entire row/col
print(b[:1])      # this assumes the starting index of the row is 0, so we select the entire first row, prints [[1, 2, 3]]
```

### Array Manipulation

Array manipulation is particularly useful when certain formulas in later chapters require building matrices or perform certain operations on matrices (for instance, transposing a matrix). Below are some examples:

```python
import numpy as np
a = np.array([3, 7, 9, 10, 3, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])

# we can reshape an array as long as the new shape has the same number of elements as the original shape
b = np.reshape(b, (6, 1))
a = np.reshape(a, (3, 2))

# we can flatten an array so that all the elements are collapsed into one dimension
a = a.flatten()

# we can get the transpose of a NumPy array via the following command
a_transpose = a.T
```

### Array Mathematics

NumPy provides several math functions that can be performed on each element in the array. Rather than iterating through each element, these functions will do the operation over the entire array and are hence extremely convenient. Which math functions might be relevant in Numerical Methods? Let's take a look: 

```python
import numpy as np
a = np.array([[8, 9]])
b = np.array([[1, 2, 3], [4, 5, 6]]) 

# The most important operation you'll do is matrix multiplication, we can do this easily in 2 ways
c = np.dot(a, b)
# or
c = a @ b

# We can do a lot of other operations
d = np.sin(a)
e = np.cos(a)
f = np.exp(a)
g = np.sum(a)
h = np.mean(a)
i = np.min(a)
```

### Linear Algebra

This class will often ask you to perform linear algebra operations on vectors/matrices. This is when the `numpy.linalg` library and the following functions it provides will be useful:

```python
import numpy.linalg as la

# To take the matrix inverse of a matrix A: 
matrix_inv = la.inv(A)

# To get the eigenvalues/eigenvectors of a matrix A:
eigval, eigvec = la.eig(A)

# To calculate the norm of a vector or a matrix A:
vec_norm = la.norm(A)   # can specify what type of norm as an additional param

# To solve linear equations for an equation Ax = b
x = la.solve(A, b)
```


### Random Numbers

Random numbers are always an integral part of Numerical Methods. NumPy has provided several functions that makes it super easy to use random numbers, and this will be key during chapters like Monte Carlo. Let's dive into what NumPy has to offer for random numbers: 

```python
import numpy as np
# Generating random numbers from 0 to 1:
a = np.random.rand(3, 2)    # creates a 3 x 2 array that is populated with random nums from 0 to 1

# Generating a random integer from 0 to 100:
b = np.random.randint(100)

# Generate a random value based on an array of values:
c = np.random.choice([1, 2, 3, 4])      # will randomly return one of the values within the array
```

Python also has a random module that is separate from NumPy but can be used to do a lot of similar operations as the ones introduced above. 

### Broadcasting

Broadcasting is a powerful technique in Python that allows us to perform arithmetic between two differently-shaped arrays. 

Say we have a smaller array **A** (with a shape of 1 x 5) and a larger array **B** (with a shape of 4 x 5), and we want to add these arrays together. 

Without broadcasting, only the first row in **B** would be modified by **A**. With broadcasting, however, **A**'s values are added to each row of **B**. Hence, **A** is broadcasted onto **B**. The dimension sizes need to cooperate as they did in this example, or we'll receive some error when performing this arithmetic. 

Here is an example illustrating the concept:
```python
import numpy as np

A = np.array([[1, 2, 3, 4, 5]])
B = np.array([
    [10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100],
    [110, 120, 130, 140, 150],
    [160, 170, 180, 190, 200]
])

# Add A to B using broadcasting
C = B + A

# C = np.array([[ 11,  22,  33,  44,  55],
#        [ 61,  72,  83,  94, 105],
#        [111, 122, 133, 144, 155],
#        [161, 172, 183, 194, 205]])

print(C)
```

This result demonstrates how the values from array A were added to each row of array B, thanks to broadcasting. The dimension of A (1 x 5) was compatible with B (4 x 5), allowing A to be "stretched" across B to perform the element-wise addition. ​​


## External Links
Here is some documentation for other packages we will be using in CS 357.

- [SciPy, a scientific computing package used to solve mathematical problems.](https://docs.scipy.org/doc/scipy/reference/)

- [MatPlotLib, a visualizaion and graphing package.](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot)