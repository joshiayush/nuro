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
"""Common kernel implementations for Support Vector Machines."""

import numpy as np


def grbf(x: np.ndarray, x_prime: np.ndarray, sigma: np.float32) -> np.ndarray:
  """Implements Gaussian Radial Basis Function (RBF).

  The Gaussian RBF is a commonly used kernel function in Support Vector Machines
  and other machine learning algorithms. It's defined as:

  $$K(x, x') = e^{-\\dfrac{||x - x'||^{2}}{2\\sigma^{2}}}$$

  Here's the derivation:

  * The Gaussian RBF is defined as a function of two data points, $x$ and
  $x'$, with a parameter $\\sigma$ that controls the width of the
  kernel.
  * We start by computing the euclidean distance between the two data points
  $x$ and $x'$:

  $$||x - x'||^{2} = \\sum_{i=1}^{n}(x_{i} - x'_{i})^2$$

  Now, let's insert this distance into the Gaussian RBF formula:

  $$K(x, x') = e^{-\\dfrac{||x - x'||^{2}}{2\\sigma^{2}}}$$

  We have an exponential term, and we can simplify it further:

  $$e^{-\\dfrac{||x - x'||^{2}}{2\\sigma^{2}}} =
  e^{-\\dfrac{1}{2\\sigma^{2}} \\sum_{i=1}^{n}(x_{i} - x'_{i})^2}$$

  We can further generalize it using the property
  $e^{a + b}$ = $e^a * e ^b$, we can separate the exponential
  factors:

  $$K(x, x') = e^{-\\dfrac{1}{2\\sigma^{2}}
  \\sum_{i=1}^{n}(x_{i} - x'_{i})^2} =
  \\prod_{i=1}^{n}e^{-\\dfrac{1}{2\\sigma^{2}}(x - x')^{2}}$$

  This is the mathematical implementation of the Gaussian RBF. It measures the
  similarity between two data points $x$ and $x'$ based on the
  Euclidean distance between them, with the parameter $\\sigma$
  controlling the width of the kernel.

  Args:
    x: The numpy vector $x$.
    x_prime: The numpy vector $x'$.
    sigma: The paramter $\\sigma$ to control the width of the kernel.

  Returns:
    RBF value.

  Examples:

  >>> # Example data points
  >>> x1 = np.array([1, 2, 3])
  >>> x2 = np.array([4, 5, 6])
  ...
  >>> # Set the width parameter
  >>> sigma = 1.0
  ...
  >>> # Calculate the RBF value
  >>> rbf_result = grbf(x1, x2, sigma)
  >>> print("Gaussian RBF Value:", rbf_result)
  """
  sqr_dist = np.sum(np.power((x - x_prime), 2))
  grbf_val = np.exp(-sqr_dist / (2 * sigma**2))
  return grbf_val
