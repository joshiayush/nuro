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
# pylint: disable=missing-class-docstring, inconsistent-quotes

from typing import Union

import numpy as np


class GaussianNB:
  """Gaussian Naive Bayes (GaussianNB).

  Naive Bayes methods are a set of supervised learning algorithms based on
  applying Bayes’ theorem with the “naive” assumption of conditional
  independence between every pair of features given the value of the class
  variable. Bayes theorem states the following relationship, given class
  variable `y` and dependent feature vector $x_{1}$ through $x_{n}$:

  $$P(y | x_{1}, ..., x_{n}) = \\dfrac{P(y) \\cdot
  P(x_{1}, ..., x_{n} | y)}{P(x_{1}, ..., x_{n})}$$

  Using the naive conditional independence assumption that:

  $$P(x_{i} | y, x_{1}, ..., x_{i-1}, x_{i+1}, ..., x_{n}) = P(x_{i} | y)$$

  for all $i$, this relationship is simplified to:

  $$P(y | x_{1}, ..., x_{n}) = \\dfrac{P(y) \\cdot
  \\prod_{i=1}^{n}P(x_{i} | y)}{P(x_{1}, ..., x_{n})}$$

  Since $P(x_{1}, ..., x_{n})$ is constant given the input, we can use the
  following classification rule:

  $$P(y | x_{1}, ..., x_{n}) \\propto P(y) \\cdot \\prod_{i=1}^{n}P(x_{i} | y)$$

  $$⇒ \\hat y = arg \\max_{y} P(y) \\cdot \\prod_{i=1}^{n} P(x_{i} | y)$$

  Note:

    We never multiply probabilities in computer science since, the number can
    multiply to $0$ upto the machine precision. It's better to use the
    monotonic function $\\mathrm{log}$ and add the log of the
    probabilities.

  $$⇒ \\hat y = arg \\max_{y} \\sum_{i=1}^{n}
  \\mathrm{log}(P(x_{i} | y)) + \\mathrm{log}(P(y))$$

  and we can use Maximum A Posteriori (MAP) estimation to estimate $P(y)$
  and $P(x_{i} | y)$;  the former is then the relative frequency of class
  $y$ in the training set.

  `GaussianNB` implements the Gaussian Naive Bayes algorithm for classification.
  The likelihood of the features is assumed to be Gaussian:

  $$P(x_{i} | y) = \\dfrac{1}{\\sqrt{2 \\pi \\sigma_{y}^{2}}}
  \\exp \\left( -\\dfrac{(x_{i} - \\mu_{y})^2}{
  2 \\sigma_{y}^2} \\right)$$

  The parameters $\\sigma_{y}$ and $\\mu_{y}$ are estimated using
  maximum likelihood.
  """
  _parameter_constraints: dict = {
    'priors': ['list', 'ndarray'],
  }

  @staticmethod
  def _check_if_parameters_comply_to_constraints(**kwargs: dict) -> None:
    """Private static method to ensure the compatibility of the hyperparameters
    passed to the `GaussianNB`.

    Args:
      kwargs: Passed hyperparameters.

    Raises:
      ValueError: If any hyperparameter is not compatible.
    """
    if kwargs['priors'] is not None and kwargs[
      'priors'
    ].__class__.__name__ not in GaussianNB._parameter_constraints[
      'priors']:
      raise ValueError(
        f'"{kwargs["priors"].__class__.__name__}" type prior is'
        ' not supported'
      )

  def __init__(self, *, priors: Union[list, np.ndarray] = None):
    """Initializes model's parameters.

    Args:
      priors: Prior probabilities of the classes. If specified, the priors are
        not adjusted according to the data.
    """
    self._priors = priors
    self._mean = None
    self._var = None

    self._check_if_parameters_comply_to_constraints(priors=self._priors)

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
    """Fit Gaussian Naive Bayes according to X, y.

    Args:
      X: Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
      y: Target values.

    Returns:
      Returns the instance itself.
    """
    self._classes = np.unique(y)

    n_samples, n_features = X.shape
    n_classes = len(self._classes)
    self._var = np.zeros((n_classes, n_features))
    self._mean = np.zeros((n_classes, n_features))

    # Take into account the priors that were initialised early
    if self._priors is not None:
      priors = np.asarray(self._priors)
      # Check that the provided priors matches the number of classes
      if len(priors) != n_classes:
        raise ValueError('Number of priors must match number of classes.')
      # Check that all the priors add to 1.0
      if not np.isclose(priors.sum(), 1.0):
        raise ValueError('The sum of the priors should be 1.')
      # Check that the priors are non-negative
      if (priors < 0).any():
        raise ValueError('Priors must be non-negative.')
      self._class_priors = priors
    else:
      self._class_priors = np.zeros(n_classes, dtype=np.float64)

    for idx, c in enumerate(self._classes):
      X_c = X[y == c]
      self._mean[idx, :] = X_c.mean(axis=0)
      self._var[idx, :] = X_c.var(axis=0)

      # Update if only no priors is provided
      if self._priors is None:
        self._class_priors[idx] = X_c.shape[0] / float(n_samples)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict for `X` using the previously calculated priors.

    Args:
      X: Testing vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    Returns:
      Predictions made for the given testing vector `X`.
    """
    if self._mean is None or self._var is None:
      raise RuntimeError(
        f'{self.__class__.__name__}: predict called before fitting data'
      )

    if X.shape[1] != self._mean.shape[1]:
      raise ValueError(
        (
          f'Number of features {X.shape[1]} does not match previous data '
          f'{self._mean.shape[1]}.'
        )
      )

    preds = []
    for x in X:
      posteriors = []
      for idx, _ in enumerate(self._classes):
        # Since, the computational precision might erase the bits while
        # multiplying extremely small integers (i.e., zeros), we compute the
        # summation of the log of the probability of each `x` given the class
        # `y` and also add the prior rather multiplying it
        posterior = np.sum(np.log(self._pdf(idx, x))
                           ) + np.log(self._class_priors[idx])
        posteriors.append(posterior)
      # Only add classes with the highest posterior
      preds.append(self._classes[np.argsort(posteriors)])
    return np.array(preds)

  def _pdf(self, c_idx: int, x: np.ndarray) -> np.float64:
    """Probability density function to compute the following for the given `x`
    given the class i.e., the `c_idx`:

    $$P(x_{i} | y) = \\dfrac{1}{\\sqrt{2 \\pi \\sigma_{y}^{2}}}
    \\exp \\left( -\\dfrac{(x_{i} - \\mu_{y})^2}{
    2 \\sigma_{y}^2} \\right)$$

    Args:
      c_idx: Index of the class for which the Probability density function is
        called.
      x: The variable whose probability needs to be calculated given the class.

    Returns:
      The probability of `x` given the class `y`.
    """
    return np.exp(
      -(np.power((x - self._mean[c_idx]), 2) / 2 * self._var[c_idx])
    ) / np.sqrt(2 * np.pi * self._var[c_idx])
