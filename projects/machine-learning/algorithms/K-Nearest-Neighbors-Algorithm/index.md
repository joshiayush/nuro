# K-Nearest Neighbors Algorithm


The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.

For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. While this is technically considered “plurality voting”, the term, “majority vote” is more commonly used in literature. The distinction between these terminologies is that “majority voting” technically requires a majority of greater than 50%, which primarily works when there are only two categories. When you have multiple classes—e.g. four categories, you don’t necessarily need 50% of the vote to make a conclusion about a class; you could assign a class label with a vote of greater than 25%.

<div align="left">

<img height="400px" src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/ef/3a/KNN.component.complex-narrative-xl-retina.ts=1653407890466.png/content/adobe-cms/us/en/topics/knn/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative/items/content_group/image" />

</div>

Regression problems use a similar concept as classification problem, but in this case, the average the k nearest neighbors is taken to make a prediction about a classification. The main distinction here is that classification is used for discrete values, whereas regression is used with continuous ones. However, before a classification can be made, the distance must be defined. Euclidean distance is most commonly used, which we’ll delve into more below.

It's also worth noting that the KNN algorithm is also part of a family of “lazy learning” models, meaning that it only stores a training dataset versus undergoing a training stage. This also means that all the computation occurs when a classification or prediction is being made. Since it heavily relies on memory to store all its training data, it is also referred to as an instance-based or memory-based learning method.

## Compute KNN: distance metrics

To recap, the goal of the k-nearest neighbor algorithm is to identify the nearest neighbors of a given query point, so that we can assign a class label to that point. In order to do this, KNN has a few requirements:

__Determine your distance metrics__

In order to determine which data points are closest to a given query point, the distance between the query point and the other data points will need to be calculated. These distance metrics help to form decision boundaries, which partitions query points into different regions. You commonly will see decision boundaries visualized with Voronoi diagrams.

__Euclidean distance (p=2):__ This is the most commonly used distance measure, and it is limited to real-valued vectors. Using the below formula, it measures a straight line between the query point and the other point being measured.

$\mathrm{Eculidean\ Distance} = d(x, y) = \sqrt{\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}$

__Manhattan distance (p=1):__ This is also another popular distance metric, which measures the absolute value between two points. It is also referred to as taxicab distance or city block distance as it is commonly visualized with a grid, illustrating how one might navigate from one address to another via city streets.

$\mathrm{Manhattan\ Distance} = d(x, y) = \left( \sum_{i=1}^{m}|x_{i} - y_{i}| \right)$

__Minkowski distance:__ This distance measure is the generalized form of Euclidean and Manhattan distance metrics. The parameter, p, in the formula below, allows for the creation of other distance metrics. Euclidean distance is represented by this formula when p is equal to two, and Manhattan distance is denoted with p equal to one.

$\mathrm{Minkowski\ Distance} = d(x, y) = \left( \sum_{i=1}^{n}|x_{i} - y_{i}| \right)^{\dfrac{1}{p}}$

__Hamming distance:__ This technique is used typically used with Boolean or string vectors, identifying the points where the vectors do not match. As a result, it has also been referred to as the overlap metric. This can be represented with the following formula:

$\mathrm{Hamming\ Distance} = D_{H} = \left( \sum_{i=1}^{k}|x_{i} - y_{i}| \right)$

$x = y, D = 0$

$x \neq y, D \neq 1$

As an example, if you had the following strings, the hamming distance would be 2 since only two of the values differ.

<div align="center">

<img width="800px" height="300px" src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/e8/5a/HammingDistanceExample.component.complex-narrative-xl-retina.ts=1653407891607.png/content/adobe-cms/us/en/topics/knn/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative0/items/content_group_1457609298/image" />

</div>

## Compute KNN: defining k


The k value in the k-NN algorithm defines how many neighbors will be checked to determine the classification of a specific query point. For example, if k=1, the instance will be assigned to the same class as its single nearest neighbor. Defining k can be a balancing act as different values can lead to overfitting or underfitting. Lower values of k can have high variance, but low bias, and larger values of k may lead to high bias and lower variance. The choice of k will largely depend on the input data as data with more outliers or noise will likely perform better with higher values of k. Overall, it is recommended to have an odd number for k to avoid ties in classification, and cross-validation tactics can help you choose the optimal k for your dataset.

```python
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
```

```python
_MAGIC_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/ai/projects/datasets/magic+gamma+telescope/magic04.data'
```

We need to explicitly define names since the dataset does not come with a column field.

```python
df = pd.read_csv(
    _MAGIC_DATASET_PATH,
    names=[
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class",
    ],
)
```

```python
df
```

###### Output


```
fLength    fWidth   fSize   fConc  fConc1     fAsym   fM3Long  \
0       28.7967   16.0021  2.6449  0.3918  0.1982   27.7004   22.0110   
1       31.6036   11.7235  2.5185  0.5303  0.3773   26.2722   23.8238   
2      162.0520  136.0310  4.0612  0.0374  0.0187  116.7410  -64.8580   
3       23.8172    9.5728  2.3385  0.6147  0.3922   27.2107   -6.4633   
4       75.1362   30.9205  3.1611  0.3168  0.1832   -5.5277   28.5525   
...         ...       ...     ...     ...     ...       ...       ...   
19015   21.3846   10.9170  2.6161  0.5857  0.3934   15.2618   11.5245   
19016   28.9452    6.7020  2.2672  0.5351  0.2784   37.0816   13.1853   
19017   75.4455   47.5305  3.4483  0.1417  0.0549   -9.3561   41.0562   
19018  120.5135   76.9018  3.9939  0.0944  0.0683    5.8043  -93.5224   
19019  187.1814   53.0014  3.2093  0.2876  0.1539 -167.3125 -168.4558   

       fM3Trans   fAlpha     fDist class  
0       -8.2027  40.0920   81.8828     g  
1       -9.9574   6.3609  205.2610     g  
2      -45.2160  76.9600  256.7880     g  
3       -7.1513  10.4490  116.7370     g  
4       21.8393   4.6480  356.4620     g  
...         ...      ...       ...   ...  
19015    2.8766   2.4229  106.8258     h  
19016   -2.9632  86.7975  247.4560     h  
19017   -9.4662  30.2987  256.5166     h  
19018  -63.8389  84.6874  408.3166     h  
19019   31.4755  52.7310  272.3174     h  

[19020 rows x 11 columns]
```

```python
df['class'].unique()
```

###### Output


```
array(['g', 'h'], dtype=object)
```

Since, the class labels only contains letters and we need numerical data to establish a mapping from our independent variables to our dependent variables - we need to convert the classes into numerical data.

```python
df['class'] = (df['class'] == 'g').astype(int)
```

Now we're all set with our labels.

```python
df['class'].unique()
```

###### Output


```
array([1, 0])
```

Let's plot histograms of our labels.

```python
for label in df.columns[:-1]:
    plt.figure(figsize=(2, 2))
    plt.hist(
        df[df["class"] == 1][label],
        color="blue",
        label="gamma",
        alpha=0.7,
        density=True,
    )
    plt.hist(
        df[df["class"] == 0][label],
        color="red",
        label="hedron",
        alpha=0.5,
        density=True,
    )
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
```

###### Output


## Train, Validation, and Test Split

```python
train, valid, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.7 * len(df)), int(0.83 * len(df))]
)
```

Let's rebalance our training set so that the proportion of `alpha` and `hedron` particles remain same.

```python
def ScaleDataset(df, over_sample=False):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if over_sample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y
```

```python
train, X_train, y_train = ScaleDataset(train, over_sample=True)
```

```python
valid, X_valid, y_valid = ScaleDataset(valid)
```

```python
test, X_test, y_test = ScaleDataset(test)
```

```python
sum(y_train == 1), sum(y_train == 0)
```

###### Output


```
(8650, 8650)
```

## Training our model

```python
model = KNeighborsClassifier(n_neighbors=25, metric='minkowski', p=2)
model.fit(X_train, y_train)
```

###### Output


```
KNeighborsClassifier(n_neighbors=25)
```

Making predictions from our trained model.

```python
y_pred = model.predict(X_test)
```

```python
print(classification_report(y_test, y_pred))
```

###### Output


```
precision    recall  f1-score   support

           0       0.79      0.71      0.74      1126
           1       0.85      0.90      0.87      2108

    accuracy                           0.83      3234
   macro avg       0.82      0.80      0.81      3234
weighted avg       0.83      0.83      0.83      3234
```

```python
joblib.dump(model, 'kNN_magic_model.sav')
```

###### Output


```
['kNN_magic_model.sav']
```