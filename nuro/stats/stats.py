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
# pylint: disable=missing-class-docstring, protected-access
"""An implementation of statistical operations for `ai`.

Note:

  We don't support `numpy` arrays of more than 2 dimensions but plan to do it in
  future.

  Also, unlike `numpy` functions our implementation assumes the given dataset to
  be a sample not a population hence uses `ddof=1`.
"""

from typing import Optional

import numpy as np

from . import _core


def mean(a: np.ndarray, /, *, axis: Optional[int] = None) -> np.ndarray:
  """Compute the mean along the specified axis.

  Returns the average of the array elements. The average is taken over the
  flattened array by default, otherwise over the specified axis.

  $$\\bar x = \\dfrac{\\sum_{i=1}^{n}x_{i}}{n}$$

  Args:
    a: A 2-D numpy vector.
    axis: A integer value specifying along which axis to calculate the mean.

  Returns:
    The mean for each vector.

  Raises:
    `ValueError` if `a` has more than 1 dimension.

  Note:

    The arithmetic mean is the sum of the elements along the axis divided by the
    number of elements.

    Note that for floating-point input, the mean is computed using the same
    precision the input has.

  Example:

      >>> import numpy as np
      >>> a = np.array([[1, 2], [3, 4]])
      >>> a
      array([[1, 2],
      [3, 4]])
      >>> from ai import stats
      >>> stats.mean(a)
      array([2.5])
      >>> stats.mean(a, axis=0)
      array([1.5, 3.5])
      >>> stats.mean(a, axis=1)
      array([2., 3.])

      In single precision, `mean` can be inaccurate:

      >>> a = np.zeros((2, 512*512), dtype=np.float32)
      >>> a[0, :] = 1.0
      >>> a[1, :] = 0.1
      >>> a
      array([[1. , 1. , 1. , ..., 1. , 1. , 1. ],
      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1]], dtype=float32)
      >>> stats.mean(a)
      array([0.54999924])

  """
  return _core.mean(a, axis=axis)


def median(a: np.ndarray, /, *, axis: Optional[int] = None) -> np.ndarray:
  """Compute the median along the specified axis.

  The median is often compared with other descriptive statistics such as the
  `mean` (average), `mode`, and `std` (standard deviation) and is robust to
  outliers.

  For odd number of numbers in a dataset the median is calculated using:

  $$M = \\dfrac{n+1}{2}; \\quad \\text{odd}$$

  For even number of numbers in a dataset the median is calculated using:

  $$M = \\dfrac{1}{2}(\\dfrac{n}{2} + \\dfrac{n+1}{2}); \\quad \\text{even}$$


  Args:
    a: A 2-D numpy vector.
    axis: A integer value specifying along which axis to calculate the mean.

  Returns:
    The median for each vector.

  Note:

    Given a vector `V` of length `N`, the median of `V` is the middle value of a
    sorted copy of `V`, `V_sorted` - i e., `V_sorted[(N-1)/2]`, when `N` is odd,
    and the average of the two middle values of `V_sorted` when `N` is even.

  Example:

      >>> import numpy as np
      >>> x = np.random.random((3, 3))
      >>> x
      array([[0.1676058 , 0.21633727, 0.12763747],
      [0.36879157, 0.45505013, 0.06045118],
      [0.88213891, 0.95437981, 0.61791297]])
      >>> from ai import stats
      >>> stats.median(x)
      array([0.36879157])
      >>> stats.median(x, axis=0)
      array([0.1676058 , 0.36879157, 0.88213891])
      >>> stats.median(x, axis=1)
      array([0.36879157, 0.45505013, 0.12763747])

  """
  return _core.median(a, axis=axis)


def std(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  """Compute the standard deviation along the specified axis.

  Returns the standard deviation, a measure of the spread of a distribution, of
  the array elements. The standard deviation is computed for the flattened array
  by default, otherwise over the specified axis.

  $$s = \\sqrt{\\dfrac{\\sum_{i=1}^{n}(x_{i} - \\bar x)^{2}}{n - 1}}$$

  Args:
    a: Calculate the standard deviation of these values.
    ddof: Means Delta Degrees of Freedom. The divisor used in calculations is
      `N - ddof`, where `N` represents the number of elements. By default `ddof`
      is one.
    axis: Axis along which the standard deviation is computed. The default is to
      compute the standard deviation of the flattened array.

  Returns:
    Return a new array containing the standard deviation.

  Note:

    The standard deviation is the square root of the average of the squared
    deviations from the `mean`, i.e., `std = sqrt(mean(x))`, where
    `x = abs(a - a.mean())**2`.

    The average squared deviation is typically calculated as `x.sum() / N`,
    where `N = len(x)`. If, however, `ddof` is specified, the divisor `N - ddof`
    is used instead. In standard statistical practice, `ddof=1` provides an
    unbiased estimator of the variance of the infinite population. `ddof=0`
    provides a maximum likelihood estimate of the variance for normally
    distributed variables. The standard deviation computed in this function is
    the square root of the estimated variance, so even with `ddof=1`, it will
    not be an unbiased estimate of the standard deviation per se.

  Example:

      >>> import numpy as np
      >>> a = np.array([[1, 2], [3, 4]])
      >>> a
      array([[1, 2],
      [3, 4]])
      >>> from ai import stats
      >>> stats.std(a)
      array([1.29099445])
      >>> stats.std(a, axis=0)
      array([0.70710678, 0.70710678])
      >>> stats.std(a, axis=1)
      array([1.41421356, 1.41421356])

      In single precision, `std` can be inaccurate:

      >>> a = np.zeros((2, 512*512), dtype=np.float32)
      >>> a[0, :] = 1.0
      >>> a[1, :] = 0.1
      >>> a
      array([[1. , 1. , 1. , ..., 1. , 1. , 1. ],
      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1]], dtype=float32)
      >>> stats.std(a)
      array([0.45000043])

  """
  return _core.std(a, ddof=ddof, axis=axis)


def var(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  """Compute the variance along the specified axis.

  Returns the variance of the array elements, a measure of the spread of a
  distribution. The variance is computed for the flattened array by default,
  otherwise over the specified axis.

  Args:
    a: Array containing numbers whose variance is desired.
    ddof: Means Delta Degrees of Freedom. The divisor used in calculations is
      `N - ddof`, where `N` represents the number of elements. By default `ddof`
      is one.
    axis: Axis along which the variance is computed. The default is to compute
      the variance of the flattened array.

  Returns:
    A new array containing the variance.

  $$s^{2} = \\dfrac{\\sum_{i=1}^{n}(x_{i} - \\bar x)^{2}}{n - 1}$$

  Note:

    The variance is the average of the squared deviations from the `mean`, i.e.,
    `var = mean(x)`, where `x = abs(a - a.mean())**2`.

    The mean is typically calculated as `x.sum() / N`, where `N = len(x)`. If,
    however, `ddof` is specified, the divisor `N - ddof` is used instead. In
    standard statistical practice, `ddof=1` provides an unbiased estimator of
    the variance of a hypothetical infinite population. `ddof=0` provides a
    maximum likelihood estimate of the variance for normally distributed
    variables.

  Example:

      >>> import numpy as np
      >>> a = np.array([[1, 2], [3, 4]])
      >>> a
      array([[1, 2],
      [3, 4]])
      >>> from ai import stats
      >>> stats.var(a)
      array([1.66666667])
      >>> stats.var(a, axis=0)
      array([0.5, 0.5])
      >>> stats.var(a, axis=1)
      array([2., 2.])

      In single precision, `var` can be inaccurate:

      >>> a = np.zeros((2, 512*512), dtype=np.float32)
      >>> a[0, :] = 1.0
      >>> a[1, :] = 0.1
      >>> a
      array([[1. , 1. , 1. , ..., 1. , 1. , 1. ],
      [0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1]], dtype=float32)
      >>> stats.var(a)
      array([0.20250039])

  """
  return _core.var(a, ddof=ddof, axis=axis)


def zscore(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  """Compute the z-score along the specified axis.

  Compute the `z` score of each value in the sample, relative to the sample mean
  and standard deviation.

  $$z = \\dfrac{x - \\bar x}{s}$$

  Args:
    a: An array like object containing the sample data.
    ddof: Means Delta Degrees of Freedom. The divisor used in calculations is
      `N - ddof`, where `N` represents the number of elements. By default `ddof`
      is one.
    axis: Axis along which the z-score is computed. The default is to compute
      the z-score of the flattened array.

  Returns:
    A new array containing the z-scores.

  Note:

    The `z-score` for a data value is its difference from the relative `mean`
    divided by the relative `std` (standard deviation), i.e.,
    `z = x - x.mean() / x.std()`.

  Example:

      >>> import numpy as np
      >>> a = np.array([ 0.7972,  0.0767,  0.4383,  0.7866,  0.8091,
      ...                0.1954,  0.6307,  0.6599,  0.1065,  0.0508])
      >>> a
      array([0.7972, 0.0767, 0.4383, 0.7866, 0.8091, 0.1954, 0.6307, 0.6599,
      0.1065, 0.0508])
      >>> from ai import stats
      >>> stats.zscore(a)
      array([[ 1.06939901, -1.1830039 , -0.05258212,  1.03626165,  1.10660039,
      -0.81192795,  0.5488923 ,  0.64017636, -1.08984414, -1.26397161]])

      Computing along a specified axis.

      >>> b = np.array([[ 0.3148,  0.0478,  0.6243,  0.4608],
      ...               [ 0.7149,  0.0775,  0.6072,  0.9656],
      ...               [ 0.6341,  0.1403,  0.9759,  0.4064],
      ...               [ 0.5918,  0.6948,  0.904 ,  0.3721],
      ...               [ 0.0921,  0.2481,  0.1188,  0.1366]])
      >>> b
      array([[0.3148, 0.0478, 0.6243, 0.4608],
      [0.7149, 0.0775, 0.6072, 0.9656],
      [0.6341, 0.1403, 0.9759, 0.4064],
      [0.5918, 0.6948, 0.904 , 0.3721],
      [0.0921, 0.2481, 0.1188, 0.1366]])
      >>> stats.zscore(b, axis=0)
      array([[-0.19264823, -1.28415119,  1.07259584,  0.40420358],
      [ 0.33048416, -1.37380874,  0.04251374,  1.00081084],
      [ 0.26796377, -1.12598418,  1.23283094, -0.37481053],
      [-0.22095197,  0.24468594,  1.19042819, -1.21416216],
      [-0.82780366,  1.4457416 , -0.43867764, -0.1792603 ]])
      >>> stats.zscore(b, axis=1)
      array([[-0.59710641,  0.94678835,  0.63499955,  0.47177349, -1.45645498],
      [-0.73263586, -0.62041675, -0.3831319 ,  1.71200261,  0.0241819 ],
      [-0.0644368 , -0.11512076,  0.97769651,  0.76458677, -1.56272573],
      [-0.02464405,  1.63406492, -0.20339557, -0.31610104, -1.08992426]])

  """
  return _core.zscore(a, ddof=ddof, axis=axis)


def varcoef(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  """Compute the coefficients of variation along the specified axis.

  The coefficient of variation (CV) is the ratio of the standard deviation to
  the mean. The higher the coefficient of variation, the greater the level of
  dispersion around the mean.

  $$cv = \\dfrac{s}{\\bar x} * 100 \\%$$

  Args:
    a: An array like object containing the sample data.
    ddof: Means Delta Degrees of Freedom. The divisor used in calculations is
      `N - ddof`, where `N` represents the number of elements. By default `ddof`
      is one.
    axis: Axis along which the coefficient of variation is computed. The default
      is to compute the coefficient of variation of the flattened array.

  Returns:
    A new array containing the coefficients of variation.

  Note:

    The coefficient of variation is the ratio of the relative `std`
    (standard deviation) over the `mean` and is represented as percentage.

  Example:

      >>> import numpy as np
      >>> a = np.array([[1, 2], [3, 4]])
      >>> a
      array([[1, 2],
      [3, 4]])
      >>> from ai import stats
      >>> stats.varcoef(a)
      array([51.63977795])
      >>> stats.varcoef(a, axis=0)
      array([47.14045208, 20.20305089])
      >>> stats.varcoef(a, axis=1)
      array([70.71067812, 47.14045208])

  """
  return _core.varcoef(a, ddof=ddof, axis=axis)


def cov(
  a: np.ndarray,
  b: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  """Compute the relationship between `a` and `b` along the specified axis.

  Covariance measures the total variation of two random variables from their
  expected values.

  $$cov_{x,y} = \\dfrac{\\sum_{i=1}^{N}(x_{i}-\\bar x)(y_{i}-\\bar y)}{N-1}$$

  Args:
    a: An array like object containing the sample data.
    ddof: Means Delta Degrees of Freedom. The divisor used in calculations is
      `N - ddof`, where `N` represents the number of elements. By default `ddof`
      is one.
    axis: Axis along which the covariance is computed. The default is to compute
      the covariance of the flattened array.

  Returns:
    A new array containing the covariance.

  Note:

    * Positive covariance indicates that the two variables tend to move in the
      same direction.
    * Negative covariance indicates that the two variables tend to move in
      inverse direction.

  Example:

      >>> import numpy as np
      >>> x, y = np.random.random((3, 3)), np.random.random((3, 3))
      >>> x
      array([[0.62809713, 0.81040891, 0.16158262],
      [0.82474163, 0.08633899, 0.60068869],
      [0.55120899, 0.31197217, 0.05694431]])
      >>> y
      array([[0.31343184, 0.54189237, 0.5759936 ],
      [0.47156163, 0.07193879, 0.88730511],
      [0.673533  , 0.28599424, 0.90187499]])
      >>> from ai import stats
      >>> stats.cov(x, y)
      array([-0.99412023])
      >>> stats.cov(x, y, axis=0)
      array([-1.0114404 , -0.93096473, -1.01969051])
      >>> stats.cov(x, y, axis=1)
      array([-1.00575977, -0.94265263, -0.98948036])

  """
  return _core.cov(a, b, ddof=ddof, axis=axis)


def corrcoef(
  a: np.ndarray,
  b: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  return _core.corrcoef(a, b, ddof=ddof, axis=axis)
