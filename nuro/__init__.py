# pylint: disable=empty-docstring
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

from typing import Union, List

import os
import pathlib
import base64

from bs4 import BeautifulSoup
from urllib.parse import urlparse

from . import stats
from . import boolalg
from . import neighbors
from . import naive_bayes
from . import linear_model

_DOCS_DIR = pathlib.Path(
  os.fspath(os.path.normpath(__file__))
).parent.parent / 'docs'
_ML_DOCS_DIR = _DOCS_DIR / 'ml'


def _ConvertImagePath2Base64(soup: BeautifulSoup) -> BeautifulSoup:
  for img in soup.find_all('img'):
    src = img.get('src')

    if urlparse(src).scheme == '':
      with open(src, 'rb') as img_f:
        base64_img = base64.b64encode(img_f.read()).decode('utf-8')
      img['src'] = f'data:image/png;base64,{base64_img}'
  return soup


def _WrapImgStrongWithBlock(soup: BeautifulSoup) -> BeautifulSoup:
  for img in soup.find_all('img'):
    if img.parent.find_all('strong'):
      img.parent.strong.wrap(soup.new_tag('p'))
  return soup


def _MarkdownPreprocessPipeline(
  soup: BeautifulSoup, pipeline: List[callable]
) -> BeautifulSoup:
  for call in pipeline:
    soup = call(soup)
  return soup


def _PreprocessReadme(fpath: Union[str, pathlib.Path]) -> str:
  with open(fpath, mode='r', encoding='utf-8') as f:
    readme = f.read()

  soup = BeautifulSoup(readme, 'html.parser')
  soup = _MarkdownPreprocessPipeline(
    soup, (_ConvertImagePath2Base64, _WrapImgStrongWithBlock)
  )
  return str(soup)


# DO NOT TOUCH! THIS KEEPS THE DOCUMENTATION IN ORDER!
_ML_README_FILES = (
  'Introduction-to-ML.md',
  'Descending-into-ML.md',
  'Reducing-Loss.md',
  'Generalization.md',
  'Training-and-Test-Sets.md',
  'Validation-Set.md',
  'Representation.md',
  'Feature-Crosses.md',
  'Transform-Your-Data.md',
  'Regularization-for-Simplicity.md',
  'Logistic-Regression.md',
  'Classification.md',
  'Regularization-for-Sparsity.md',
  'Neural-Networks.md',
  'Training-Neural-Networks.md',
)


def _MakeDocs(docs_dir: pathlib.Path) -> str:  # pylint: disable=unused-argument
  doc = ''
  for readme in _ML_README_FILES:
    doc += _PreprocessReadme(_ML_DOCS_DIR / readme)
  return doc


__doc__ = _MakeDocs(_ML_DOCS_DIR)
