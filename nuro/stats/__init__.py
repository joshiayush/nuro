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
"""Statistical routines for `numpy` arrays.

`ai.stats` implements variety of statistical methods for computing stats of
`numpy` arrays:

##### Averages and Variances

  * `ai.stats.mean`
  * `ai.stats.median`
  * `ai.stats.std`
  * `ai.stats.var`
  * `ai.stats.zscore`
  * `ai.stats.varcoef`

##### Correlating

  * `ai.stats.cov`
  * `ai.stats.corrcoef`
"""

from .stats import mean
from .stats import median
from .stats import std
from .stats import var
from .stats import zscore
from .stats import varcoef
from .stats import cov
from .stats import corrcoef
