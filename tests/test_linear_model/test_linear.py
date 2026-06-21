"""Tests for linear regression model."""
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

from numpy.testing import (
  assert_array_almost_equal,
)

from nuro.linear_model import LinearRegression


def test_linear_regression():
  # Test LinearRegression on a simple dataset; a simple dataset
  X = np.array([[1], [2]])  # pylint: disable=invalid-name
  y = np.array([1, 2])

  model = LinearRegression()
  model.fit(X, y)

  # although scikit-learn api tests for upto 8 decimal points we only test for 1
  assert_array_almost_equal(model.predict(X), [1.0, 2.0], decimal=1)

  # also testing for degenerate input
  X = np.array([[1]])  # pylint: disable=invalid-name
  y = [0]

  model = LinearRegression()
  model.fit(X, y)

  assert_array_almost_equal(model.predict(X), [0], decimal=1)
