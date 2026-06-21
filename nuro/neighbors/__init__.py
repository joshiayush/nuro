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

"""`ai.neighbors` provides functionality for `unsupervised` and `supervised`
neighbors-based learning methods. Unsupervised nearest neighbors is the
foundation of many other learning methods, notably manifold learning and
spectral clustering. Supervised neighbors-based learning comes in two flavors:
`classification` for data with discrete labels, and `regression` for data with
continuous labels.

The principle behind nearest neighbor methods is to find a predefined number of
training samples closest in distance to the new point, and predict the label
from these. The number of samples can be a user-defined constant
`(k-nearest neighbor learning)`, or vary based on the local density of points
`(radius-based neighbor learning)`. The distance can, in general, be any metric
measure: standard `Euclidean` distance is the most common choice.
Neighbors-based methods are known as non-generalizing machine learning methods,
since they simply “remember” all of its training data (possibly transformed into
a fast indexing structure such as a `Ball Tree` or `KD Tree`).

Despite its simplicity, nearest neighbors has been successful in a large number
of classification and regression problems, including handwritten digits and
satellite image scenes. Being a non-parametric method, it is often successful in
classification situations where the decision boundary is very irregular.

`ai.neighbors` implements the following nearest-neighbors algorithms:

  * `ai.neighbors.knn.KNeighborsClassifier`
"""

from .knn import KNeighborsClassifier
