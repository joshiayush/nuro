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

"""An implementation of Pearson's correlation coefficient algorithm."""

import warnings

import numpy as np


def cov(  # pylint: disable=too-many-positional-arguments
  m: np.ndarray,
  y: np.ndarray = None,
  rowvar: bool = True,
  bias: bool = False,
  ddof: int = None,
  fweights: np.ndarray = None,
  aweights: np.ndarray = None,
  *,
  dtype: np.dtype = None
) -> np.ndarray:
  """Estimate a covariance matrix, given data and weights.

  Covariance indicates the level to which two variables vary together.
  If we examine N-dimensional samples, $X = [x_1, x_2, ... x_N]^T$, then
  the covariance matrix element $C_{ij}$ is the covariance of $x_i$
  and $x_j$. The element $C_{ii}$ is the variance of $x_i$.

  Args:
    m: A 1-D or 2-D array containing multiple variables and observations. Each
      row of `m` represents a variable, and each column a single observation of
      all those variables. Also see `rowvar` below.
    y: An additional set of variables and observations. `y` has the same form as
      that of `m`.
    rowvar: If `rowvar` is True (default), then each row represents a variable,
      with observations in the columns. Otherwise, relationship is transposed:
      each column represents a variable, while the rows contains observations.
    bias: Default normalization (False) is by ``(N - 1)``, where ``N`` is the
      number of observations given (unbiased estimate). If `bias` is True, then
      normalization is by ``N``. These values can be overriden by using the
      keyword ``ddof``.
    ddof: If not ``None`` then the default value implied by `bias` is
      overridden. Note that ``ddof=1`` will return the unbiased estimate, even
      if both `fweights` and `aweights` are specified, and ``ddof=0`` will
      return the simple average.
    fweights: 1-D array of integer frequency weight; the number of times each
      observation vector should be repeated.
    aweights: 1-D array of observation vector weights. These relative weights
      are typically large for observations considered "important" and smaller
      for observations considered less "important". If ``ddof=0`` the array of
      weights can be used to assign probabilities to observation vectors.
    dtype: Data type of the result. By default the return data type will have at
      least `numpy.float64` precision.

  Returns:
    The covariance matrix of the variables.

  Note:

      Assume that the observations are in the columns of the observation array
      `m` and let ``f=fweights`` and ``a=aweights`` for brevity. The steps to
      compute the weighted covariance are as follows:

      >>> m = np.arange(10, dtype=np.float64)
      >>> f = np.arange(10) * 2
      >>> a = np.arange(10) ** 2
      >>> ddof = 1
      >>> w = f * a
      >>> v1 = np.sum(w)
      >>> v2 = np.sum(w * a)
      >>> m -= np.sum(m * w, axis=None, keepdims=True) / v1
      >>> cov = np.dot(m * w, m.T) * v1 / ((v1 ** 2) - (ddof * v2))

      Note that when ``a == 1``, the normalization factor
      ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)`` as it
      should.
  """
  if ddof is not None and ddof != int(ddof):
    raise ValueError('ddof must be integer')

  # Handling complex arrays too.
  m = np.asarray(m)
  if m.ndim > 2:
    raise ValueError('m has more than 2 dimensions')

  if y is not None:
    y = np.asarray(y)
    if y.ndim > 2:
      raise ValueError('y has more than 2 dimensions')

  if dtype is None:
    if y is None:
      dtype = np.result_type(m, np.float64)
    else:
      dtype = np.result_type(m, y, np.float64)

  x = np.array(m, ndmin=2, dtype=dtype)
  if not rowvar and x.shape[0] != 1:
    x = x.T
  if x.shape[0] == 0:
    return np.array([]).reshape(0, 0)
  if y is not None:
    y = np.asarray(y)
    if not rowvar and y.shape[0] != 1:
      y = y.T
    x = np.concatenate((x, y), axis=0)

  if ddof is None:
    if bias == 0:
      ddof = 1
    else:
      ddof = 0

  # Get the product of frequencies and weights.
  w = None
  if fweights is not None:
    fweights = np.asarray(fweights, dtype=float)
    if not np.all(fweights == np.around(fweights)):
      raise TypeError('fweights must be integer')
    if fweights.ndim > 1:
      raise RuntimeError('Cannot handle multidimensional fweights')
    if fweights.shape[0] != x.shape[1]:
      raise RuntimeError('Incompatible number of samples and fweights')
    if any(fweights < 0):
      raise ValueError('fweights cannot be negative')
    w = fweights

  if aweights is not None:
    aweights = np.asarray(aweights, dtype=float)
    if aweights.ndim > 1:
      raise RuntimeError('Cannot handle multidimensional aweights')
    if aweights.shape[0] != x.shape[1]:
      raise RuntimeError('Incompatible number of samples and fweights')
    if any(aweights < 0):
      raise ValueError('aweights cannot be negative')
    if w is None:
      w = aweights
    else:
      w *= aweights

  avg, w_sum = np.average(x, axis=1, weights=w, returned=True)
  w_sum = w_sum[0]

  # Determine the normalization.
  if w is None:
    fact = x.shape[1] - ddof
  elif ddof == 0:
    fact = w_sum
  elif aweights is None:
    fact = w_sum - ddof
  else:
    fact = w_sum - ddof * sum(w * aweights) / w_sum

  if fact <= 0:
    warnings.warn(
      'Degrees of freedom <= 0 for slice', RuntimeWarning, stacklevel=3
    )
    fact = 0.0

  x -= avg[:, None]
  xtrans = None
  if w is None:
    xtrans = x.T
  else:
    xtrans = (x * w).T
  c = np.dot(x, xtrans.conj())
  c *= np.true_divide(1, fact)
  return c.squeeze()


def corrcoef(
  x: np.ndarray,
  y: np.ndarray = None,
  rowvar: bool = True,
  *,
  dtype: np.dtype = None
) -> np.ndarray:
  """Return Pearson product-moment correlation coefficient.

  The relationship between the correlation coefficient matrix, `R`, and the
  covariance matrix, `C`, is

  $$R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }$$

  The values of `R` are between -1 and 1, inclusive.

  Args:
    m: A 1-D or 2-D array containing multiple variables and observations. Each
      row of `m` represents a variable, and each column a single observation of
      all those variables. Also see `rowvar` below.
    y: An additional set of variables and observations. `y` has the same form as
      that of `m`.
    rowvar: If `rowvar` is True (default), then each row represents a variable,
      with observations in the columns. Otherwise, relationship is transposed:
      each column represents a variable, while the rows contains observations.
    dtype: Data type of the result. By default the return data type will have at
      least `numpy.float64` precision.

  Returns:
    The correlation coefficient matrix of the variables.
  """
  c = cov(x, y, rowvar, dtype=dtype)
  try:
    d = np.diag(c)
  except ValueError:
    # Scalar covariance; NaN if incorrect value (NaN, Inf, 0), 1 otherwise.
    return c / c
  stddev = np.sqrt(d.real)
  c /= stddev[:, None]
  c /= stddev[None, :]

  # Clip real and imaginary parts to [-1, 1]. This does not guarantee
  # abs([i,j]) <= 1 for complex arrays, but is the best we can do without
  # excessive work.
  np.clip(c.real, -1, 1, out=c.real)
  if np.iscomplexobj(c):
    np.clip(c.imag, -1, 1, out=c.imag)
  return c
