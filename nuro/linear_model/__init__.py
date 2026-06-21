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

"""The term linear model implies that the model is specified as a linear
combination of features. Based on training data, the learning process computes
one weight for each feature to form a model that can predict or estimate the
target value.

`ai.linear_model` module implements a variety of linear models:

  * `ai.linear_model.linear.LinearRegression`
  * `ai.linear_model.logistic.LogisticRegression`
  * `ai.linear_model.perceptron.Perceptron`
"""

from .linear import LinearRegression
from .logistic import LogisticRegression
from .perceptron import Perceptron
