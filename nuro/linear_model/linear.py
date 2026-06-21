# Copyright 2023 The AI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-function-args, invalid-name, missing-module-docstring
# pylint: disable=missing-class-docstring
"""A numpy-compatible linear regression implementation.

Example:

    ```python
    import numpy as np

    from sklearn import datasets
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    from ai.linear_model import LinearRegression

    X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
    )
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred, y_test))
    ```
"""

import numpy as np


class LinearRegression:
  """`LinearRegression` fits a linear model with coefficients w = (w1, ..., wp)
  to minimize the residual sum of squares between the observed targets in the
  dataset, and the targets predicted by the linear approximation.
  """
  def __init__(self, *, alpha: np.float16 = .01, n_iters: int = 1000):
    """Initializes model's `learning rate` and number of `iterations`.

    Args:
      alpha: Model's learning rate. High value might over shoot the minimum
        loss, while low values might make the model to take forever to learn.
      n_iters: Maximum number of updations to make over the weights and bias in
        order to reach to a effecient prediction that minimizes the loss.
    """
    self._alpha = alpha
    self._n_iters = n_iters
    self._bias = None
    self._weights = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
    """Fit the linear model on `X` given `y`.

    Hypothesis function for our `LinearRegression` $\\hat y = b + wX$,
    where `b` is the model's intercept and `w` is the coefficient of `X`.

    The cost function or the loss function that we use is the Mean Squared Error
    (MSE) between the predicted value and the true value. The cost function
    `(J)` can be written as:

    $$J = \\dfrac{1}{m}\\sum_{i=1}^{n}(\\hat y_{i} - y_{i})^2$$

    To achieve the best-fit regression line, the model aims to predict the
    target value $\\hat Y$ such that the error difference between the
    predicted value $\\hat Y$ and the true value $Y$ is minimum. So,
    it is very important to update the `b` and `w` values, to reach the best
    value that minimizes the error between the predicted `y` value and the true
    `y` value.

    A linear regression model can be trained using the optimization algorithm
    gradient descent by iteratively modifying the model’s parameters to reduce
    the mean squared error (MSE) of the model on a training dataset. To update
    `b` and `w` values in order to reduce the Cost function (minimizing RMSE
    value) and achieve the best-fit line the model uses Gradient Descent. The
    idea is to start with random `b` and `w` values and then iteratively update
    the values, reaching minimum cost.

    On differentiating cost function `J` with respect to `b`:

    $$\\dfrac{dJ}{db} = \\dfrac{2}{n} \\cdot
    \\sum_{i=1}^{n}(\\hat y_{i} - y_{i})$$

    On differentiating cost function `J` with respect to `w`:

    $$\\dfrac{dJ}{dw} = \\dfrac{2}{n} \\cdot
    \\sum_{i=1}^{n}(\\hat y_{i} - y_{i}) \\cdot x_{i}$$

    The above derivative functions are used for updating `weights` and `bias` in
    each iteration.

    Args:
      X: Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
      y: Target vector.
    """
    n_samples, n_features = X.shape
    self._bias = 0
    self._weights = np.zeros(n_features)

    for _ in range(self._n_iters):
      y_pred = np.dot(X, self._weights) + self._bias

      weights_d = (1 / n_samples) * np.dot(X.T, (y_pred - y))
      bias_d = (1 / n_samples) * np.sum((y_pred - y))

      self._weights = self._weights - (self._alpha * weights_d)
      self._bias = self._bias - (self._alpha * bias_d)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict for `X` using the previously calculated `weights` and `bias`.

    Args:
      X: Feature vector.

    Returns:
      Target vector.

    Raises:
      RuntimeError: If `predict` is called before `fit`.
      ValueError: If shape of the given `X` differs from the shape of the `X`
        given to the `fit` function.
    """
    if self._weights is None or self._bias is None:
      raise RuntimeError(
        f'{self.__class__.__name__}: predict called before fitting data'
      )

    if X.shape[1] != self._weights.shape[0]:
      raise ValueError(
        (
          f'Number of features {X.shape[1]} does not match previous data '
          f'{self._weights.shape[0]}.'
        )
      )

    return np.dot(X, self._weights) + self._bias
