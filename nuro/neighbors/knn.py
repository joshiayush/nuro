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

from typing import Union

import numpy as np

from collections import Counter


class DistanceMetric:
  """Distance metrices for computing k-nearest neighbors.

  These distance metrices can be used for computing the distances between two
  `np.ndarray` and not just for `kNeighborsClassifier`. This `DistanceMetric`
  class supports `Euclidean`, `Minkowski`, `Manhattan`, and `Hamming` distance
  metrices to compute the distance between two data points.

  Args:
    metric: The metric to use for computing distance. Default `minkowski`.
    minkowski_p: Power parameter for the Minkowski metric.
  """
  _distance_func_cache = None

  def __init__(self, metric: str = 'minkowski', minkowski_p: int = 2):
    """Initializes metric parameters.

    Args:
      metric: The metric to use for computing distance. Default `minkowski`.
      minkowski_p: Power parameter for the Minkowski metric.
    """
    self._metric = metric
    self._minkowski_p = minkowski_p

  def euclidean(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    """Euclidean distance.

    This is the most commonly used distance measure, and it is limited to
    real-valued vectors. Using the below formula, it measures a straight line
    between the query point and the other point being measured.

    $$\\mathrm{Euclidean\\ Distance} =
    \\sqrt{\\sum_{i=1}^{n}(y_{i} - x_{i})^2}$$

    Args:
      x1: Query point vector.
      x2: Other point vector.

    Returns:
      Measured distance point vector.
    """
    return np.sqrt(np.sum(np.power((x1 - x2)), 2))

  def minkowski(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    """Minkowski distance.

    This distance measure is the generalized form of Euclidean and Manhattan
    distance metrics. The parameter, p, in the formula below, allows for the
    creation of other distance metrics. Euclidean distance is represented by
    this formula when p is equal to two, and Manhattan distance is denoted with
    p equal to one.

    $$\\mathrm{Minkowski\\ Distance} =
    ((\\sum_{i=1}^{n}|x_{i} - y_{i}|) ^ \\dfrac{1}{p})$$

    Args:
      x1: Query point vector.
      x2: Other point vector.

    Returns:
      Measured distance point vector.
    """
    return np.power(np.sum(np.absolute((x1 - x2))), 1 / self._minkowski_p)

  def manhattan(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    """Manhattan distance.

    This is also another popular distance metric, which measures the absolute
    value between two points. It is also referred to as taxicab distance or city
    block distance as it is commonly visualized with a grid, illustrating how
    one might navigate from one address to another via city streets.

    $$\\mathrm{Manhattan\\ Distance} = \\sum_{i=1}^{m}|x_{i} - y_{i}|$$

    Args:
      x1: Query point vector.
      x2: Other point vector.

    Returns:
      Measured distance point vector.
    """
    return np.sum(np.absolute((x1 - x2)))

  def hamming(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    """Hamming distance.

    This technique is typically used with Boolean or string vectors, identifying
    the points where the vectors do not match. As a result, it has also been
    referred to as the overlap metric. This can be represented with the
    following formula:

    $$\\mathrm{Hamming\\ Distance} = \\sum_{i=1}^{k}|x_{i} - y_{i}|$$

    Args:
      x1: Query point vector.
      x2: Other point vector.

    Returns:
      Measured distance point vector.
    """
    return np.sum(np.absolute((x1 - x2)))

  def distance(
    self, x1: Union[np.float32, np.ndarray], x2: Union[np.float32, np.ndarray]
  ) -> Union[np.float32, np.ndarray]:
    """Distance function that uses one of `euclidean`, `minkowski`, `manhattan`,
    or `hamming` distance metric to measure distance between the given points.

    Args:
      x1: Query point vector.
      x2: Other point vector.

    Returns:
      Measured distance point vector.
    """
    if self._distance_func_cache is not None:
      return self._distance_func_cache(x1, x2)

    if self._metric == 'euclidean':
      self._distance_func_cache = self.euclidean
    elif self._metric == 'minkowski':
      self._distance_func_cache = self.minkowski
    elif self._metric == 'manhattan':
      self._distance_func_cache = self.manhattan
    elif self._metric == 'hamming':
      self._distance_func_cache = self.hamming
    else:
      raise RuntimeError(
        (
          f'{self.__class__.__name__}: {self._metric} is not one of '
          '["euclidean", "minkowski", "manhattan", "hamming"]'
        )
      )
    return self._distance_func_cache(x1, x2)


class KNeighborsClassifier(DistanceMetric):
  """Classifier implementing the k-nearest neighbors vote.

  The k-nearest neighbors algorithm is a non-parametric, supervised learning
  classifier, which uses proximity to make classifications or predictions about
  the grouping of an individual data point. k-nearest neighbors algorithm is
  typically used as a classification algorithm, working off the assumption that
  similar points can be found near one another.

  For classification problems, a class label is assigned on the basis of a
  majority vote - i.e., the label that is most frequently represented around a
  give data point is used.

  Args:
    n_neighbors: Number of neighbors to use. By default `3`.
    p: Power parameter for the Minkowski metric. When `p = 1`, this is
      equivalent to using `manhattan_distance (l1)`, and
      `euclidean_distance (l2)` for
      `p = 2`. For arbitrary `p`, `minkowski_distance (l_p)` is used.
    metric: Metric to use for distance computation. Default is `euclidean`.
  """
  _parameter_constraints: dict = {
    'metric': [
      ('euclidean', 'supported'), ('minkowski', 'not-supported'),
      ('manhattan', 'not-supported'), ('hamming', 'not-supported')
    ]
  }

  @staticmethod
  def _check_if_parameters_comply_to_constraints(**kwargs: dict) -> None:
    """Private static method to ensure the compatibility of the hyperparameters
    passed to the `KNeighborsClassifier`.

    Args:
      kwargs: Passed hyperparameters.

    Raises:
      ValueError: If any hyperparameter is not compatible.
    """
    for (metric_name, metric_status
         ) in KNeighborsClassifier._parameter_constraints['metric']:
      if kwargs['metric'] == metric_name:
        if metric_status != 'supported':
          raise ValueError(
            f'distance metric {metric_name} is not supported yet'
          )
        break

  def __init__(
    self, *, n_neighbors: int = 3, p: int = 2, metric: str = 'euclidean'
  ):
    """Initializes model's hyperparameters.

    Args:
      n_neighbors: Number of neighbors to use. By default `3`.
      p: Power parameter for the Minkowski metric. When `p = 1`, this is
        equivalent to using `manhattan_distance (l1)`, and
        `euclidean_distance (l2)` for `p = 2`. For arbitrary `p`,
        `minkowski_distance (l_p)` is used.
      metric: Metric to use for distance computation. Default is `euclidean`.
    """
    self._n_neighbors = n_neighbors
    self._p = p
    self._metric = metric
    self._is_fitted = False

    self._check_if_parameters_comply_to_constraints(metric=self._metric)
    super().__init__(self._metric, self._p)

  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    """Fit the k-nearest neighbors classifier from the training dataset.

    Args:
      X: Sample vector.
      y: Target vector.

    Returns:
      The fitted `KNeighborsClassifier`.
    """
    self._X = X
    self._y = y
    self._is_fitted = True
    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict the class labels for the provided data.

    Args:
      X: Test samples.

    Returns:
      Class label for each data sample.

    Raises:
      RuntimeError: If predict method is called before fit.
    """
    if self._is_fitted is False:
      raise RuntimeError(
        f'{self.__class__.__name__}: predict called before fitting data'
      )

    preds = []
    for x in X:
      # Compute the distance of the new point `x` from all the points in the
      # training set using the initialized distance metric
      distances = [self.distance(x, x_train) for x_train in self._X]

      # Extract all `_n_neighbors` distances from a sorted list of `distances`
      k_indices = np.argsort(distances)[:self._n_neighbors]
      # Finally extract the labels using the above `k_indices`
      k_nearest_labels = [self.y_train[i] for i in k_indices]

      # Calculate the prediction using "plurality voting"
      preds = [*preds, Counter(k_nearest_labels).most_common()[0][0]]
    return np.array(preds)
