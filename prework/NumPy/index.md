# NumPy

[Numpy](https://numpy.org/doc/stable/index.html) is a Python library for creating and manipulating matrices, the main data structure used by ML algorithms. [Matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)) are mathematical objects used to store values in rows and columns. 

Python calls matrices *lists*, NumPy calls them *arrays* and TensorFlow calls them *tensors*. Python represents matrices with the [list data type](https://docs.python.org/3/library/stdtypes.html#lists).

## Import NumPy module

Run the following code cell to import the NumPy module:

```python
import numpy as np
```

## Populate arrays with specific numbers

Call `np.array` to create a NumPy array with your own hand-picked values. For example, the following call to `np.array` creates an 8-element array:

```python
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)
```

###### Output


```
[1.2 2.4 3.5 4.7 6.1 7.2 8.3 9.5]
```

You can also use `np.array` to create a two-dimensional array. To create a two-dimensional array specify an extra layer of square brackets. For example, the following call creates a `3x2` array:

```python
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)
```

###### Output


```
[[ 6  5]
 [11  7]
 [ 4  8]]
```

To populate an array with all zeroes, call `np.zeros`. To populate an array with all ones, call `np.ones`.

## Populate arrays with sequences of numbers

You can populate an array with a sequence of numbers:

```python
sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)
```

###### Output


```
[ 5  6  7  8  9 10 11]
```

Notice that `np.arange` generates a sequence that includes the lower bound (5) but not the upper bound (12).

## Populate arrays with random numbers

NumPy provides various functions to populate arrays with random numbers across certain ranges. For example, `np.random.randint` generates random integers between a low and high value. The following call populates a 6-element array with random integers between 50 and 100. 

```python
random_integers_between_50_and_100 = np.random.randint(low=50, high=101,
                                                       size=(6))
print(random_integers_between_50_and_100)
```

###### Output


```
[50 52 90 58 57 66]
```

Note that the highest generated integer `np.random.randint` is one less than the `high` argument.

To create random floating-point values between 0.0 and 1.0, call `np.random.random`. For example:

```python
random_floats_between_0_and_1 = np.random.random(6)
print(random_floats_between_0_and_1) 
```

###### Output


```
[0.58412468 0.42188739 0.03088132 0.42848895 0.66435921 0.56969786]
```

## Mathematical Operations on NumPy Operands

If you want to add or subtract two arrays, linear algebra requires that the two operands have the same dimensions. Furthermore, if you want to multiply two arrays, linear algebra imposes strict rules on the dimensional compatibility of operands. Fortunately, NumPy uses a trick called [**broadcasting**](https://developers.google.com/machine-learning/glossary/#broadcasting) to virtually expand the smaller operand to dimensions compatible for linear algebra. For example, the following operation uses broadcasting to add 2.0 to the value of every item in the array created in the previous code cell:

```python
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)
```

###### Output


```
[2.58412468 2.42188739 2.03088132 2.42848895 2.66435921 2.56969786]
```

The following operation also relies on broadcasting to multiply each cell in an array by 3.0:

```python
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3.0
print(random_integers_between_150_and_300)
```

###### Output


```
[150. 156. 270. 174. 171. 198.]
```

## Task 1: Create a Linear Dataset

Your goal is to create a simple dataset consisting of a single feature and a label as follows:

1. Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named `feature`.
2. Assign 15 values to a NumPy array named `label` such that:

```
   label = (3)(feature) + 4
```
For example, the first value for `label` should be:

```
  label = (3)(6) + 4 = 22
 ```

```python
feature = np.arange(6, 21)
print(feature)
# Pure Python approach:
# label = np.array(list(map(lambda x : (3 * x) + 4, feature)))
label = (feature * 3) + 4
print(label)
```

###### Output


```
[ 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
[22 25 28 31 34 37 40 43 46 49 52 55 58 61 64]
```

## Task 2: Add Some Noise to the Dataset

To make your dataset a little more realistic, insert a little random noise into each element of the `label` array you already created. To be more precise, modify each value assigned to `label` by adding a *different* random floating-point value between -2 and +2. 

Don't rely on broadcasting. Instead, create a `noise` array having the same dimension as `label`.

```python
# A pure mathematical approach:
# noise = (np.random.random(label.size) * 4) - 2
noise = np.random.randint(low=-2, high=2, size=(label.size))
print(noise)
label = label + noise
print(label)
```

###### Output


```
[-1  1 -1 -2 -2  0  0 -1  0 -1 -2 -2 -2  0 -2]
[21 26 27 29 32 37 40 42 46 48 50 53 56 61 62]
```