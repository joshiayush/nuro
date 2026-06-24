# Naive Bayes

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable $y$ and dependent feature vector $x_{1}$ through $x_{n}$:

$$P(y | x_{1}, ..., x_{n}) = \dfrac{P(y) \cdot P(x_{1} ..., x_{n} | y)}{P(x_{1} ..., x_{n})}$$

Using the naive conditional independence assumption that

$$P(x_{i} | y, x_{1}, ..., x_{i-1}, x_{i+1}, ..., x_{n}) = P(x_{i} | y)$$

for all $i$, this relationship is simplified to

$$P(y | x_{1}, ..., x_{n}) = \dfrac{P(y) \cdot \prod_{i=1}^{n}P(x_{i} | y) }{P(x_{1} ..., x_{n})}$$

Since $P(x_{1}, ..., x_{n})$ is constant given the input, we can use the following classification rule:

$$P(y | x_{1}, ..., x_{n}) \propto P(y) \cdot \prod_{i=1}^{n}P(x_{i} | y)$$

$$⇒ \hat y = arg \max_{y} P(y) \cdot \prod_{i=1}^{n} P(x_{i} | y)$$

and we can use __Maximum A Posteriori (MAP)__ estimation to estimate $P(y)$ and $P(x_{i} | y)$;  the former is then the relative frequency of class $y$ in the training set.

## Gaussian Naive Bayes

[__GaussianNB__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian:

$$P(x_{i} | y) = \dfrac{1}{\sqrt{2 \pi \sigma_{y}^{2}}} \exp \left( -\dfrac{(x_{i} - \mu_{y})^2}{2 \sigma_{y}^2} \right)$$

The parameters $\sigma_{y}$ and $\mu_{y}$ are estimated using maximum likelihood.

```python
import joblib

from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
```

```python
X, y = load_iris(return_X_y=True)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
model = GaussianNB()
model.fit(X_train, y_train)
```

###### Output


```
GaussianNB()
```

```python
y_pred = model.predict(X_test)
```

```python
print(classification_report(y_test, y_pred))
```

###### Output


```
precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

```python
joblib.dump(model, 'naive_bayes_iris_model.sav')
```

###### Output


```
['naive_bayes_iris_model.sav']
```