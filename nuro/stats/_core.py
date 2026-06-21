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

from typing import Optional

import numpy as np


def _validate_array_dims(func) -> callable:
  def wrapper(*args, **kwargs):
    for arg in args:
      if isinstance(arg, np.ndarray) and arg.ndim > 2:
        raise ValueError(
          f'{func.__name__} does not support vector of more than 2 dimensions.'
        )

    if kwargs['axis'] and kwargs['axis'] > 2:
      raise ValueError(
        f'{func.__name__} does not support vector of more than 2 dimensions.'
      )
    return func(*args, **kwargs)

  return wrapper


def _transpose_vector_if_required(func) -> callable:
  def wrapper(*args, **kwargs):
    args_t = tuple()
    for arg in args:
      if isinstance(arg, np.ndarray):
        if 'axis' in kwargs and kwargs['axis'] is None:
          args_t += (np.ravel(arg), )
        elif ('axis' in kwargs and kwargs['axis'] == 1):
          args_t += (arg.T, )
        else:
          args_t += (arg, )
    return func(*args_t, **kwargs)

  return wrapper


def _mean(x: np.ndarray) -> np.float64:
  return np.sum(x) / x.size


@_validate_array_dims
@_transpose_vector_if_required
def mean(a: np.ndarray, /, *, axis: Optional[int] = None) -> np.ndarray:
  mean_val = []
  if axis is None:
    mean_val.append(_mean(a))
  else:
    for x in a:
      mean_val.append(_mean(x))
  return np.array(mean_val)


def _median(x: np.ndarray) -> np.float64:
  x_sorted = np.sort(x, kind='stable')

  counts = x_sorted.size
  h = counts // 2

  # duplicate high if odd number of elements so mean does nothing
  odd = counts % 2 == 1
  l = np.where(odd, h, h - 1)
  return np.divide(x_sorted[l] + x_sorted[h], 2.)


@_validate_array_dims
@_transpose_vector_if_required
def median(a: np.ndarray, /, *, axis: Optional[int] = None) -> np.ndarray:
  median_val = []
  if axis is None:
    median_val.append(_median(a))
  else:
    for x in a:
      median_val.append(_median(x))
  return np.array(median_val)


def _std(x: np.ndarray, /, *, ddof: Optional[int] = 1) -> np.float64:
  _n = x.size
  return np.sqrt(
    np.divide(
      _n * np.sum(np.power(x, 2)) - np.power(np.sum(x), 2), _n * (_n - ddof)
    )
  )


@_validate_array_dims
@_transpose_vector_if_required
def std(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  std_val = []
  if ddof not in (
    0,
    1,
  ):
    raise ValueError(f'ddof must be in (0, 1), you gave ddof={ddof}')
  if axis is None:
    std_val.append(_std(a, ddof=ddof))
  else:
    for x in a:
      std_val.append(_std(x, ddof=ddof))
  return np.array(std_val)


@_validate_array_dims
@_transpose_vector_if_required
def var(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  return np.power(std(a, ddof=ddof, axis=axis), 2)


def _zscore(
  x: np.ndarray, /, *, mean_val: np.float64, std_val: np.float64
) -> np.float64:
  return np.divide(x - mean_val, std_val)


@_validate_array_dims
@_transpose_vector_if_required
def zscore(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  zscore_val = []
  a_mean = mean(a, axis=axis)
  a_std = std(a, ddof=ddof, axis=axis)
  if axis is None:
    zscore_val.append(_zscore(a, mean_val=a_mean, std_val=a_std))
  else:
    for idx, x in enumerate(a):
      zscore_val.append(_zscore(x, mean_val=a_mean[idx], std_val=a_std[idx]))
  return np.array(zscore_val)


@_validate_array_dims
@_transpose_vector_if_required
def varcoef(
  a: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  return np.divide(std(a, ddof=ddof, axis=axis), mean(a, axis=axis)) * 100


def _cov(
  x: np.ndarray,
  y: np.ndarray,
  /,
  *,
  x_mean: np.float64,
  y_mean: np.float64,
  ddof: Optional[int] = 1
) -> np.float64:
  _n = x.size
  return np.sum(np.multiply((x - x_mean), (y - y_mean))) / _n - ddof


@_validate_array_dims
@_transpose_vector_if_required
def cov(
  a: np.ndarray,
  b: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if a.size != b.size:
    raise ValueError(
      f"given vectors aren't of equal length a.size={a.size} != b.size={b.size}"
    )
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  cov_val = []
  a_mean = mean(a, axis=axis)
  b_mean = mean(b, axis=axis)
  if axis is None:
    cov_val.append(_cov(a, b, x_mean=a_mean, y_mean=b_mean, ddof=ddof))
  else:
    for idx, (ai, bi) in enumerate(zip(a, b)):
      cov_val.append(
        _cov(ai, bi, x_mean=a_mean[idx], y_mean=b_mean[idx], ddof=ddof)
      )
  return np.array(cov_val)


cov._require_2d = True  # pylint: disable=protected-access


def _corrcoef(
  *, cov_val: np.float64, a_std: np.float64, b_std: np.float64
) -> np.float64:
  return cov_val / a_std * b_std


@_validate_array_dims
@_transpose_vector_if_required
def corrcoef(
  a: np.ndarray,
  b: np.ndarray,
  /,
  *,
  ddof: Optional[int] = 1,
  axis: Optional[int] = None
) -> np.ndarray:
  if a.size != b.size:
    raise ValueError(
      f"given vectors aren't of equal length a.size={a.size} != b.size={b.size}"
    )
  if axis == 1:
    # if array is already transposed we prevent it from further transpose in the
    # next call since (X_T)_T = X
    axis = 0
  corrcoef_val = []
  a_std = std(a, ddof=ddof, axis=axis)
  b_std = std(b, ddof=ddof, axis=axis)
  cov_val = cov(a, b, ddof=ddof, axis=axis)
  if axis is None:
    corrcoef_val.append(
      _corrcoef(cov_val=cov_val, a_std=a_std, b_std=b_std)
    )
  else:
    for idx in range(len(a)):
      corrcoef_val.append(
        _corrcoef(cov_val=cov_val[idx], a_std=a_std[idx], b_std=b_std[idx])
      )
  return np.array(corrcoef_val)
