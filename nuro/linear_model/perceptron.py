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

import numpy as np


class Perceptron:
  """Perceptron classifier.

  The perceptron is a simple supervised machine learning algorithm used to
  classify data into binary outcomes. It is a type of linear classifier, i.e. a
  classification algorithm that makes its predictions based on a linear
  predictor function combining a set of weights with the feature vector.

  $$y = \\begin{cases} 1 & \\text{if } w \\cdot x + b > 0 \\\\
  0 & \\text{otherwise} \\end{cases}$$
  """

  def __init__(self,
               *,
               alpha: np.float16 = np.float16(.01),
               n_iters: np.int64 = np.int64(1000),
               random_state: int = 1):
    """Initializes model's `learning rate` and number of `iterations`.

    Args:
      alpha: Model's learning rate. High value might over shoot
        the minimum loss, while low values might make the model to take forever
        to learn.
      n_iters: Maximum number of updations to make over the weights and bias in
        order to reach to a effecient prediction that minimizes the loss.
      random_state: Seed to generate random weights and bias.
    """
    self.alpha = alpha
    self.n_iters = n_iters
    self.random_state = random_state

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
    """Fit training data.
    
    Args:
      X: Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
      y: Target values.
    
    Returns:
      self: An instance of self.
    """
    rgen = np.random.RandomState(self.random_state)
    self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
    self.b_ = np.float64(0.)
    self.errors_ = []
    for _ in range(self.n_iters):
      errors = 0
      for xi, target in zip(X, y):
        update = self.alpha * (target - self.predict(xi))
        self.weights += update * xi
        self.b_ += update
        errors += int(update != 0.0)
      self.errors_.append(errors)
    return self

  def net_input(self, X):
    """Calculate net input.
    
    Args:
      X: Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
        
    Returns:
      The dot product of `X` and `weights` plus `b`.
    """
    return np.dot(X, self.weights) + self.b_

  def predict(self, X):
    """Return class label after unit step.
    
    Args:
      X: Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    
    Returns:
      The class label after unit step.
    """
    return np.where(self.net_input(X) >= 0.0, 1, 0)
