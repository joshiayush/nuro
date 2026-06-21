"""Tests for correlation statistics."""
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

import numpy as np

from nuro.stats import correlation


def test_cov_1d_array():
  data = np.array([1, 2, 3, 4, 5])
  result = correlation.cov(data)
  expected = np.var(data, ddof=1)
  assert np.isclose(result, expected)


def test_cov_2d_array():
  data = np.array([[1, 2, 3], [4, 5, 6]])
  result = correlation.cov(data)
  expected = np.cov(data)
  assert np.allclose(result, expected)


def test_corrcoef_1d_array():
  data = np.array([1, 2, 3, 4, 5])
  result = correlation.corrcoef(data)
  expected = np.corrcoef(data)
  assert np.allclose(result, expected)


def test_corrcoef_2d_array():
  data = np.array([[1, 2, 3], [4, 5, 6]])
  result = correlation.corrcoef(data)
  expected = np.corrcoef(data)
  assert np.allclose(result, expected)


def test_corrcoef_with_weights():
  data = np.array([[1, 2, 3], [4, 5, 6]])
  y = np.array([[7, 8, 9], [10, 11, 12]])
  result = correlation.corrcoef(data, y=y)
  expected = np.corrcoef(data, y=y)
  assert np.allclose(result, expected)
