# Pandas DataFrame

Introduction to [**DataFrames**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), which are the central data structure in the pandas API. This is not a comprehensive DataFrames tutorial. Rather, this provides a very quick introduction to the parts of DataFrames required to do the other Colab exercises in Machine Learning Course.

A DataFrame is similar to an in-memory spreadsheet. Like a spreadsheet:

  * A DataFrame stores data in cells. 
  * A DataFrame has named columns (usually) and numbered rows.

## Import NumPy and pandas modules

Run the following code cell to import the NumPy and pandas modules. 

```python
import numpy as np
import pandas as pd
```

## Creating a DataFrame

The following code cell creates a simple DataFrame containing 10 cells organized as follows:

  * 5 rows
  * 2 columns, one named `temperature` and the other named `activity`

The following code cell instantiates a `pd.DataFrame` class to generate a DataFrame. The class takes two arguments:

  * The first argument provides the data to populate the 10 cells. The code cell calls `np.array` to generate the `5x2` NumPy array.
  * The second argument identifies the names of the two columns.

```python
# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)
```

###### Output


```
temperature  activity
0            0         3
1           10         7
2           20         9
3           30        14
4           40        15
```

## Adding a new column to a DataFrame

You may add a new column to an existing pandas DataFrame just by assigning values to a new column name. For example, the following code creates a third column named `adjusted` in `my_dataframe`:

```python
# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)
```

###### Output


```
temperature  activity  adjusted
0            0         3         5
1           10         7         9
2           20         9        11
3           30        14        16
4           40        15        17
```

## Specifying a subset of a DataFrame

Pandas provide multiples ways to isolate specific rows, columns, slices or cells in a DataFrame.

```python
print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])
```

###### Output


```
Rows #0, #1, and #2:
   temperature  activity  adjusted
0            0         3         5
1           10         7         9
2           20         9        11 

Row #2:
   temperature  activity  adjusted
2           20         9        11 

Rows #1, #2, and #3:
   temperature  activity  adjusted
1           10         7         9
2           20         9        11
3           30        14        16 

Column 'temperature':
0     0
1    10
2    20
3    30
4    40
Name: temperature, dtype: int64
```

## Task 1: Create a DataFrame

Do the following:

  1. Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named `Eleanor`,  `Chidi`, `Tahani`, and `Jason`.  Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

  2. Output the following:

     * the entire DataFrame
     * the value in the cell of row #1 of the `Eleanor` column

  3. Create a fifth column named `Janet`, which is populated with the row-by-row sums of `Tahani` and `Jason`.

To complete this task, it helps to know the NumPy basics covered in the NumPy Tutorial.

```python
# Create a Python list that holds the names of the four columns.
my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Create a 3x4 numpy array, each cell populated with a random integer.
my_data = np.random.randint(low=0, high=101, size=(3, 4))

df = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the created dataframe.
print(df)

print('\nSecond row of the Eleanor column: %d\n' % df['Eleanor'][1])

# Create a column named Janet whose contents are the sum
# of two other columns.
df['Janet'] = df['Tahani'] + df['Jason']

# Print the enhanced data frame.
print(df)
```

###### Output


```
Eleanor  Chidi  Tahani  Jason
0       28     99      79     80
1       90     96      17     20
2        1     69      70     13

Second row of the Eleanor column: 90

   Eleanor  Chidi  Tahani  Jason  Janet
0       28     99      79     80    159
1       90     96      17     20     37
2        1     69      70     13     83
```

## Copying a DataFrame

Pandas provides two different ways to duplicate a DataFrame:

* **Referencing.** If you assign a DataFrame to a new variable, any change to the DataFrame or to the new variable will be reflected in the other. 
* **Copying.** If you call the `pd.DataFrame.copy` method, you create a true independent copy.  Changes to the original DataFrame or to the copy will not be reflected in the other. 

The difference is subtle, but important.

```python
# Create a reference by assigning my_dataframe to a new variable.
print('Experiment with a reference:')
reference_to_df = df

# Print the starting value of a particular cell.
print('  Starting value of df: %d' % df['Jason'][1])
print('  Starting value of reference_to_df: %d\n' % reference_to_df['Jason'][1])

# Modify a cell in df.
df.at[1, 'Jason'] = df['Jason'][1] + 5
print('  Updated df: %d' % df['Jason'][1])
print('  Updated reference_to_df: %d\n\n' % reference_to_df['Jason'][1])

# Create a true copy of my_dataframe
print('Experiment with a true copy:')
copy_of_my_dataframe = df.copy()

# Print the starting value of a particular cell.
print('  Starting value of df: %d' % df['Eleanor'][1])
print('  Starting value of copy_of_my_dataframe: %d\n' % 
        copy_of_my_dataframe['Eleanor'][1])

# Modify a cell in df.
df.at[1, 'Eleanor'] = df['Eleanor'][1] + 3
print('  Updated df: %d' % df['Eleanor'][1])
print('  copy_of_my_dataframe does not get updated: %d' %
        copy_of_my_dataframe['Eleanor'][1])
```

###### Output


```
Experiment with a reference:
  Starting value of df: 30
  Starting value of reference_to_df: 30

  Updated df: 35
  Updated reference_to_df: 35


Experiment with a true copy:
  Starting value of df: 96
  Starting value of copy_of_my_dataframe: 96

  Updated df: 99
  copy_of_my_dataframe does not get updated: 96
```