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
"""Linear Support Vector Classification."""

from typing import Optional

import numpy as np


class LinearSVC:
  """**Linear SVC classifier** implementation.

  Linear SVMs use a linear decision boundary to separate the data points of
  different classes. When the data can be precisely linearly separated, linear
  SVMs are very suitable. This means that a single straight line (in 2D) or a
  hyperplane (in higher dimensions) can entirely divide the data points into
  their respective classes. A hyperplane that maximizes the margin between the
  classes is the decision boundary.

  The objective of training a `LinearSVC` is to minimize the norm of the weight
  vector $||w||$, which is the slope of the decision function.

  $$\\min_{{w, b}} \\dfrac{1}{2} w^{T}w = \\dfrac{1}{2} ||w||^{2}$$

  $$\\mathrm{subject\\ to}\\ t^{i} (w^{T}x^{i} + b) \\ge 1$$

  Note:

    We are minimizing $\\dfrac{1}{2}w^{T}w$, which is equal to
    $\\dfrac{1}{2}||w||^{2}$, rather than minimizing $||w||$.
    Indeed, $\\dfrac{1}{2}||w||^{2}$ has a nice and simple derivative
    (just $w$) while $||w||$ is not differentiable at $w = 0$.
    Optimization algorithms work much better on differentiable functions.

  The cost function used by **Linear SVM classifier** is the following:

  $$J(w, b) = \\dfrac{1}{2}w^{T}w +
  C\\sum_{i=1}^{m}\\max(0, 1 - t^{i}(w^{T}x^{i} + b))$$

  The loss function used is the **Hinge Loss** function that clips the value at
  $0$:

  $$\\max(0, 1 - t^{i}(w^{T}x^{i} + b))$$
  """

  def __init__(
    self,
    *,
    alpha: Optional[np.float16] = .01,
    lambda_p: Optional[np.float32] = .01,
    n_iters: Optional[np.int64] = 1_000
  ):
    """Initializes model's `alpha`, `lambda parameter` and number of
    `iterations`.

    Args:
      alpha: Model's learning rate. High value might over shoot the minimum
        loss, while low values might make the model to take forever to learn.
      lambda_p: Lambda parameter for updating weights.
      n_iters: Maximum number of updations to make over the weights and bias in
        order to reach to a effecient prediction that minimizes the loss.
    """
    self._alpha = alpha
    self._lambda_p = lambda_p
    self._n_iters = n_iters

    self._bias = None
    self._weights = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVC':  # pylint: disable=invalid-name
    """Fit `LinearSVC` according to X, y.

    The objective of training a `LinearSVC` is to minimize the norm of the
    weight vector $||w||$, which is the slope of the decision function.

    $$\\min_{{w, b}} \\dfrac{1}{2} w^{T}w = \\dfrac{1}{2} ||w||^{2}$$

    $$\\mathrm{subject\\ to}\\ t^{i} (w^{T}x^{i} + b) \\ge 1$$

    Note:

      We are minimizing $\\dfrac{1}{2}w^{T}w$, which is equal to
      $\\dfrac{1}{2}||w||^{2}$, rather than minimizing $||w||$.
      Indeed, $\\dfrac{1}{2}||w||^{2}$ has a nice and simple derivative
      (just $w$) while $||w||$ is not differentiable at
      $w = 0$. Optimization algorithms work much better on differentiable
      functions.

    The cost function used by **Linear SVM classifier** is the following:

    $$J(w, b) = \\dfrac{1}{2}w^{T}w +
    C\\sum_{i=1}^{m}\\max(0, 1 - t^{i}(w^{T}x^{i} + b))$$

    The loss function used is the **Hinge Loss** function that clips the value
    at $0$:

    $$\\max(0, 1 - t^{i}(w^{T}x^{i} + b))$$

    Args:
      X: Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
      y: Target values.

    Returns:
      Returns the instance itself.
    """
    self._bias = 0
    self._weights = np.random.rand(X.shape[1])

    # We need the decision function to be greater than 1 for all positive
    # training instances, and lower than -1 for all negative training instances.
    # If we define ``t_i = -1`` for negative instances, and ``t_i = 1`` for
    # positive instances, then we can acheive less margin violation.
    t = np.where(y <= 0, -1, 1)
    for _ in range(self._n_iters):
      for idx, x_i in enumerate(X):
        # Since, cost function is different at different decision boundaries we
        # calculate weights and bias differently.
        if t[idx] * (np.dot(x_i, self._weights) - self._bias) >= 1:
          self._weights -= self._alpha * (2 * self._lambda_p * self._weights)
        else:
          self._weights -= self._alpha * (
            2 * self._lambda_p * self._weights - np.dot(t[idx], x_i)
          )
          self._bias -= self._alpha * t[idx]
    return self

  def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
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

    return np.sign(np.dot(X, self._weights) - self._bias)
