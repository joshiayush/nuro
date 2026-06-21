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
# pylint: disable=protected-access

import pytest
import numpy as np

from nuro.linear_model import LogisticRegression


def test_predict_simple():
  model = LogisticRegression()
  model._weights = np.array([0.5, 0.5])
  model._bias = 0.1

  X = np.array([[1, 2]])
  predicted = model.predict(X)
  assert isinstance(predicted, np.ndarray)
  assert len(predicted) == 1
  assert predicted[0] in [0, 1]


def test_predict_multiple_samples():
  model = LogisticRegression()
  model._weights = np.array([0.5, 0.5])
  model._bias = 0.1

  X = np.array([[1, 2], [2, 3], [3, 4]])
  predicted = model.predict(X)
  assert isinstance(predicted, np.ndarray)
  assert len(predicted) == 3
  assert all(label in [0, 1] for label in predicted)


def test_predict_different_weights():
  model = LogisticRegression()
  model._weights = np.array([0.2, 0.8])
  model._bias = -0.1

  X = np.array([[1, 2]])
  predicted = model.predict(X)
  assert isinstance(predicted, np.ndarray)
  assert len(predicted) == 1
  assert predicted[0] in [0, 1]


def test_predict_before_fit():
  model = LogisticRegression()
  X = np.array([[1, 2]])

  with pytest.raises(RuntimeError):
    model.predict(X)


def test_predict_shape_mismatch():
  model = LogisticRegression()
  model._weights = np.array([0.5, 0.5])
  model._bias = 0.1

  X = np.array([[1, 2, 3]])
  with pytest.raises(ValueError):
    model.predict(X)
