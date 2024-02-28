---
title: Python
description: An elementary introduction to Python.
sort: 2
author:
  - CS 357 Course Staff
changelog:
  - 
    name: Arnav Aggarwal
    netid: arnava4
    date: 2024-02-27
    message: aligned notes with slides and added additional examples
---

# Python

* * *

## Learning Objectives

*   Get familiar with the language of Python 3
*   Understand the difference between mutable and immutable objects
*   Remember, the most efficient way to study a language is to practice!
*   Google is a great resource for any programming language

## Types
  Just like any other language, variables can store data of different types in Python, such as int, float, or str. We can use the type() function to find the type of an expression.

### Example

```python
a = 2
b = 3.0
c = a + b
d = 2 * a
```

What is the correct choice?

$$
\begin{flalign} 

 & \text{A) } \text{c is float, d is float} \\ 

 & \text{B) } \text{c is float, d is int} \\ 

 & \text{C) } \text{c is int, d is int} \\ 

 & \text{D) } \text{c is int, d is float} \\ 

\end{flalign}
$$

<details>
    <summary><strong>Answer</strong></summary>

<br>

\(\bf B\)

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

In this case, a and b are not separate entities, even though they are different variables. The list ```[1, 2, 3]``` is an object, and both variable names a and b are bounded to the same list. 

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

In Python, objects are divided into **mutable** and **immutable**. Mutable objects can be modified after they are first created, including lists, dictionaries, numpy arrays, etc. Immutable objects cannot be modified once they are created, including tuples, strings, floats, etc. 

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

A
```python
a = ['hello','goodbye']
b = 'hey'
a.append(b)
c = a + [b]
```

B
```python
a = ['hello','goodbye']
b = 'hey'
c = a + [b]
a += b
```

C
```python
a = ['hello','goodbye']
b = 'hey'
c = a + [b]
a.append(b)
```

<details>
    <summary><strong>Answer</strong></summary>

<br>

\(\bf C\)

<br>
<br> 

A is incorrect because c ends up having two "hey" elements in its list, and B is incorrect because \(a += b\) adds each character of "hey" as its own element. Hence, C must be the correct choice. 

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

A
```computer_science, math ['electrical', 'physics']```

B
```computer_science, math ['electrical']```

C
```computer_science ['electrical', 'physics']```

D
```computer_science ['electrical']```

<details>
    <summary><strong>Answer</strong></summary>

<br>

\(\bf C\)

<br>
<br> 

In the above code snippet, John and Tim refer to separate objects since strings are immutable in Python, and so the variable John is equal to 'computer_science'. Lists are mutable and so Julie and Anna are equal to the same list: ['electrical', 'physics']. Hence, C is the correct choice. 

</details>


## Indexing

Indexing is important for us to be able to iterate through for loops in a variety of ways in Python. Say we have an array a, the formatting for indexing follows this standard: ```a[i:j:k]```, where i is the starting index of our iteration, j is the stopping index (exclusive) of our iteration, and k is the step size.

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
The final step is to get the resulting array of b[::-1]. There is not start or stop index given, but the step size is -1. This means that we simply need to reverse the array b, as a negative step size means we start the iteration from the end of the array rather than the start. Hence, our final array will be [9, 7, 5, 3, 1], which happens to be the answer choice E. 

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

<details>
    <summary><strong>Answer</strong></summary>

<br>

\(\bf B\)

<br>
<br> 

B is the correct answer because ints in Python are immutable. This means that incrementing a by 1 and b by 2 inside the function just creates new objects for both a and b. These new objects will be restricted to the scope of this function, and as a result the variables will not be modified. 
<br>
However, lists are mutable objects and when we append values to them we're not creating new objects. We're simply updating the variables that already have global scope, and so these variables are modified. 

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


## External Links
Here are some links to some packages we will be using in CS 357.

- Documentation for [numpy](https://numpy.org/doc/stable/).

- Docementation for [scipy](https://docs.scipy.org/doc/scipy/reference/).

- Documentation for [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot).