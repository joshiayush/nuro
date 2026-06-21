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

"""**Support vector machines (SVMs)** are a set of supervised learning methods
used for classification, regression and outliers detection.

The advantages of support vector machines are:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number
	of samples.
* Uses a subset of training points in the decision function (called support
	vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision
	function. Common kernels are provided, but it is also possible to specify
	custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoid
	over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using
	an expensive five-fold cross-validation (see Scores and probabilities, below).

Types of **SVMs**:

* **Linear SVM**

 * Linear SVMs use a linear decision boundary to separate the data points of
 	different classes. When the data can be precisely linearly separated,
 	linear SVMs are very suitable. This means that a single straight line
 	(in 2D) or a hyperplane (in higher dimensions) can entirely divide the
 	data points into their respective classes. A hyperplane that maximizes the
 	margin between the classes is the decision boundary.

* **Non-Linear SVM**
		
 * Non-Linear SVM can be used to classify data when it cannot be separated
 	into two classes by a straight line (in the case of 2D). By using kernel
 	functions, nonlinear SVMs can handle nonlinearly separable data. The
 	original input data is transformed by these kernel functions into a
 	higher-dimensional feature space, where the data points can be linearly
 	separated. A linear SVM is used to locate a nonlinear decision boundary in
 	this modified space. 

The followings are the **SVMs** implementations that `ai` includes:

  * `ai.svm.classes.LinearSVC`
"""

from .classes import LinearSVC
